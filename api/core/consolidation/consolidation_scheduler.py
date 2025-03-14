# api/core/consolidation/scheduler.py
import asyncio
import sys
from datetime import datetime, time, timedelta
import pytz
import logging
from api.utils.config import get_settings, Settings
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.core.consolidation.enhanced_memory_consolidator import EnhancedMemoryConsolidator, get_consolidation_config
from api.core.consolidation.config import ConsolidationConfig

logger = logging.getLogger(__name__)

class ConsolidationScheduler:
    def __init__(
        self,
        config: ConsolidationConfig,
        pinecone_service: PineconeService,
        llm_service: LLMService,
        run_interval_hours: int = 48,  
        timezone: str = "UTC"
    ):
        self.settings = get_settings()
        self.pinecone_service = pinecone_service
        self.llm_service = llm_service
        self.run_interval_hours = run_interval_hours
        self.timezone = pytz.timezone(timezone)
        self.consolidator = None
        self.task = None
        self.config = config
        self.last_run = None
        self.memory_graph = None
        self.relationship_detector = None
        self.paused = False  # Add a flag to track if scheduling is paused
        self.pause_event = asyncio.Event()  # Event to signal scheduling state
        self.pause_event.set()  # Initially not paused
        logger.info(f"Scheduler initialized with {run_interval_hours} hour interval ({run_interval_hours/24} days)")

    async def start(self):
        """Start the scheduled consolidation task."""
        # Check if graph memory is enabled
        if hasattr(self.settings, 'enable_graph_memory') and self.settings.enable_graph_memory:
            logger.info("Graph memory is enabled, using enhanced consolidator")
            
            # Initialize graph components
            from api.core.memory.graph.memory_graph import MemoryGraph
            from api.core.memory.graph.relationship_detector import MemoryRelationshipDetector
            from api.core.consolidation.enhanced_memory_consolidator import EnhancedMemoryConsolidator
            
            self.memory_graph = MemoryGraph()
            self.relationship_detector = MemoryRelationshipDetector(self.llm_service)
            
            # Use enhanced consolidator
            self.consolidator = EnhancedMemoryConsolidator(
                config=self.config,
                pinecone_service=self.pinecone_service,
                llm_service=self.llm_service,
                memory_graph=self.memory_graph,
                relationship_detector=self.relationship_detector
            )
            logger.info("Enhanced memory consolidator initialized with graph support")
            
            # Only initialize graph if auto_populate_graph is enabled
            if hasattr(self.settings, 'auto_populate_graph') and self.settings.auto_populate_graph:
                logger.info("Auto-populating graph as configured in settings")
                asyncio.create_task(self._initialize_graph_from_existing_memories())
            else:
                logger.info("Skipping automatic graph population (use API endpoint or manual trigger to populate)")
        else:
            logger.info("Using standard consolidator without graph memory")
            # Use original consolidator
            self.consolidator = EnhancedMemoryConsolidator(
                config=self.config,
                pinecone_service=self.pinecone_service,
                llm_service=self.llm_service
            )
        
        # Set the last run time to now minus some time to ensure we don't run immediately
        # This will make the scheduler wait for the full interval before running
        self.last_run = datetime.now(self.timezone) - timedelta(hours=self.run_interval_hours/2)
        
        self.task = asyncio.create_task(self._schedule_consolidation())
        logger.info(f"Consolidation scheduler started, will run after {self.run_interval_hours/2} hours")

    async def _initialize_graph_from_existing_memories(self):
        """Initialize the graph with existing memories from Pinecone."""
        if not self.memory_graph or not self.relationship_detector:
            logger.warning("Cannot initialize graph: graph components not available")
            return
            
        try:
            logger.info("Starting initialization of graph from existing memories...")
            
            # Use GraphMemoryFactory to perform initialization
            from api.core.memory.graph.memory_factory import GraphMemoryFactory
            
            success = await GraphMemoryFactory.initialize_graph_from_existing_memories(
                memory_graph=self.memory_graph,
                relationship_detector=self.relationship_detector,
                pinecone_service=self.pinecone_service,
                batch_size=50,
                max_memories=500  # Limit to a reasonable number for initial population
            )
            
            if success:
                logger.info("Successfully initialized graph from existing memories")
                # Log graph statistics
                stats = self.memory_graph.get_graph_stats()
                logger.info(f"Memory graph statistics: {stats}")
            else:
                logger.warning("Graph initialization completed with errors")
                
        except Exception as e:
            logger.error(f"Error initializing graph from existing memories: {e}", exc_info=True)
            # Continue execution - the graph will still work, just with fewer initial connections

    async def pause(self):
        """Pause the consolidation scheduler."""
        if not self.paused:
            logger.info("Pausing consolidation scheduler")
            self.paused = True
            self.pause_event.clear()
            return True
        else:
            logger.info("Scheduler is already paused")
            return False

    async def resume(self):
        """Resume the consolidation scheduler."""
        if self.paused:
            logger.info("Resuming consolidation scheduler")
            self.paused = False
            self.pause_event.set()
            return True
        else:
            logger.info("Scheduler is already running")
            return False

    async def get_status(self):
        """Get the current status of the scheduler."""
        return {
            "paused": self.paused,
            "next_run_time": self._calculate_next_run_time()
        }

    def _calculate_next_run_time(self):
        """Calculate and return the next scheduled run time."""
        if not self.last_run:
            # If no previous run, calculate based on current time
            return datetime.now(self.timezone) + timedelta(seconds=10)  # Just a placeholder
            
        # Calculate time until next run based on last run
        return self.last_run + timedelta(hours=self.run_interval_hours)

    async def trigger_consolidation_manually(self):
        """
        Manually trigger the consolidation process.
        Returns True if successful, False otherwise.
        """
        try:
            logger.info("Manually triggered consolidation started")
            
            # Run consolidation
            await self.consolidator.consolidate_memories()
            
            # Update last run timestamp to reset the schedule
            self.last_run = datetime.now(self.timezone)
            
            logger.info("Manual consolidation complete. Next scheduled run reset to run in "
                       f"{self.run_interval_hours} hours from now")
            
            return True
        except Exception as e:
            logger.error(f"Error in manual consolidation: {e}", exc_info=True)
            return False

    async def _schedule_consolidation(self):
        """Schedule consolidation to run at specified interval."""
        while True:
            try:
                # Calculate time until next run
                now = datetime.now(self.timezone)
                next_run = self.last_run + timedelta(hours=self.run_interval_hours)
                wait_seconds = max(0, (next_run - now).total_seconds())
                
                logger.info(f"Next consolidation scheduled in {wait_seconds/3600:.2f} hours")
                
                # Wait until scheduled time
                await asyncio.sleep(wait_seconds)
                
                # Check if we're paused before running
                if self.paused:
                    logger.info("Scheduler is paused, skipping scheduled consolidation")
                    # Wait for resume signal or a short while before checking again
                    try:
                        await asyncio.wait_for(self.pause_event.wait(), timeout=600)  # Wait up to 10 minutes
                    except asyncio.TimeoutError:
                        continue
                else:
                    # Run consolidation
                    logger.info("Starting scheduled consolidation")
                    await self.consolidator.consolidate_memories()
                    self.last_run = datetime.now(self.timezone)
                    logger.info(f"Scheduled consolidation complete. Next run in {self.run_interval_hours} hours")

            except Exception as e:
                logger.error(f"Error in consolidation schedule: {e}", exc_info=True)
                # Wait an hour before retrying on error
                await asyncio.sleep(3600)

    async def populate_graph_if_needed(self):
        """Check if graph needs population and run if necessary"""
        if not self.memory_graph or self.memory_graph.graph.number_of_nodes() > 0:
            logger.info("Graph already populated or not available, skipping population")
            return False
            
        logger.info("Graph is empty, running initial population")
        return await self._initialize_graph_from_existing_memories()

    

async def main():
    """Main function to run the scheduler."""
    try:
        # Get settings
        settings = get_settings()
        logger.info("Initializing consolidation scheduler with settings")

        # Initialize services
        pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index_name=settings.pinecone_index_name
        )
        llm_service = LLMService()
        config = get_consolidation_config()

        scheduler = ConsolidationScheduler(
            config=config,
            pinecone_service=pinecone_service,
            llm_service=llm_service,
            run_interval_hours=48, 
            timezone=settings.timezone
        )

        logger.info("Starting consolidation scheduler with 7-day interval...")
        await scheduler.start()

        # Keep the process running
        while True:
            await asyncio.sleep(3600)  # Check every hour

    except Exception as e:
        logger.error(f"Fatal scheduler error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Set up logging (if not already configured elsewhere)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())