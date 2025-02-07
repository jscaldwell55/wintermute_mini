# api/core/consolidation/scheduler.py
import asyncio
import sys
from datetime import datetime, time  # Import time separately
import pytz
import logging
from api.utils.config import get_settings, Settings # Import settings
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.core.consolidation.consolidator import MemoryConsolidator, get_consolidation_config  # Corrected import
from api.core.consolidation.models import ConsolidationConfig # Import config


logger = logging.getLogger(__name__)
#No longer needed here logger.basicConfig(level=logging.INFO)

class ConsolidationScheduler:
    def __init__(
        self,
        config: ConsolidationConfig,  # Use DI for config
        pinecone_service: PineconeService,
        llm_service: LLMService,
        run_time: time = time(hour=2, minute=0),  # 2 AM default
        timezone: str = "UTC"
    ):
        self.pinecone_service = pinecone_service
        self.llm_service = llm_service
        self.run_time = run_time
        self.timezone = pytz.timezone(timezone)  # Use timezone string
        self.consolidator = None  # Initialize to None
        self.task = None  # Initialize to None
        self.config = config # Use passed config

    async def start(self):
        """Start the scheduled consolidation task."""
        # REMOVE: config = ConsolidationConfig()  # Get config from settings, not here
        self.consolidator = MemoryConsolidator(  # Use correct class name
            config=self.config, # Pass in config
            pinecone_service=self.pinecone_service,
            llm_service=self.llm_service
        )
        self.task = asyncio.create_task(self._schedule_consolidation())
        logger.info(f"Consolidation scheduler started, will run at {self.run_time}")

    async def _schedule_consolidation(self):
        """Schedule consolidation to run at specified time."""
        while True:
            try:
                # Calculate time until next run.  Correctly handle timezone.
                now = datetime.now(self.timezone)
                target_time = time(hour=self.run_time.hour, minute=self.run_time.minute) # Use a time object
                target_datetime = datetime.combine(now.date(), target_time, tzinfo=self.timezone)

                if now >= target_datetime:
                    # If we've passed the time today, schedule for tomorrow
                    target_datetime += timedelta(days=1)

                # Calculate seconds to wait
                wait_seconds = (target_datetime - now).total_seconds()
                logger.info(f"Next consolidation scheduled in {wait_seconds/3600:.2f} hours")


                # Wait until scheduled time
                await asyncio.sleep(wait_seconds)

                # Run consolidation
                logger.info("Starting scheduled consolidation")
                await self.consolidator.consolidate_memories()  # Await the consolidation
                logger.info("Scheduled consolidation complete")

            except Exception as e:
                logger.error(f"Error in consolidation schedule: {e}", exc_info=True)
                # Wait an hour before retrying on error
                await asyncio.sleep(3600)
async def main(): #put into a main function
    """Main function to run the scheduler."""
    try:
        # Get settings
        settings = get_settings()
        logger.info(f"Initializing consolidation scheduler with settings:")

        # Initialize services
        pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index_name=settings.pinecone_index_name
        )
        llm_service = LLMService()
        config = get_consolidation_config() #get config

        # Initialize scheduler
        scheduler = ConsolidationScheduler(
            config=config, #pass config
            pinecone_service=pinecone_service,
            llm_service=llm_service,
            run_time=time(hour=settings.consolidation_hour,
                         minute=settings.consolidation_minute),
            timezone=settings.timezone
        )

        logger.info("Starting consolidation scheduler...")
        await scheduler.start()

        # Keep the process running
        while True:
            await asyncio.sleep(3600)  # Check every hour

    except Exception as e:
        logger.error(f"Fatal scheduler error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up logging (if not already configured elsewhere)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())