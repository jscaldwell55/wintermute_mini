# api/core/consolidation/scheduler.py
import asyncio
import sys
from datetime import datetime, time
import pytz
import logging
logging.basicConfig(level=logging.INFO)
from api.utils.config import get_settings, Settings #Import settings
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.core.consolidation.consolidator import MemoryConsolidator  # Corrected import
from api.core.consolidation.models import ConsolidationConfig
from functools import lru_cache #import

logger = logging.getLogger(__name__)

@lru_cache()
def get_consolidation_config() -> ConsolidationConfig:
    settings = get_settings()
    return ConsolidationConfig.from_settings(settings) #Corrected

class ConsolidationScheduler:
    def __init__(
        self,
        config: ConsolidationConfig, # Use DI
        pinecone_service: PineconeService,
        llm_service: LLMService,
        run_time: time = time(hour=2, minute=0),  # 2 AM default
        timezone: str = "UTC"
    ):
        self.pinecone_service = pinecone_service
        self.llm_service = llm_service
        self.run_time = run_time
        self.timezone = pytz.timezone(timezone)
        self.consolidator = None
        self.task = None
        self.config = config # Use passed config

    async def start(self):
        """Start the scheduled consolidation task."""
        # config = ConsolidationConfig()  # REMOVE: Get config from settings
        self.consolidator = MemoryConsolidator( # Use correct class name
            config=self.config,  # Pass in config
            pinecone_service=self.pinecone_service,
            llm_service=self.llm_service
        )
        self.task = asyncio.create_task(self._schedule_consolidation())
        logger.info(f"Consolidation scheduler started, will run at {self.run_time}")

    async def _schedule_consolidation(self):
        """Schedule consolidation to run at specified time."""
        while True:
            try:
                # Calculate time until next run
                now = datetime.now(self.timezone)
                target = now.replace(
                    hour=self.run_time.hour,
                    minute=self.run_time.minute,
                    second=0,
                    microsecond=0
                )

                if now >= target:
                    # If we've passed the time today, schedule for tomorrow
                    target = target.replace(day=target.day + 1)

                # Calculate seconds to wait
                wait_seconds = (target - now).total_seconds()
                logger.info(f"Next consolidation scheduled in {wait_seconds/3600:.2f} hours")

                # Wait until scheduled time
                await asyncio.sleep(wait_seconds)

                # Run consolidation
                logger.info("Starting scheduled consolidation")
                await self.consolidator.consolidate_memories()
                logger.info("Scheduled consolidation complete")

            except Exception as e:
                logger.error(f"Error in consolidation schedule: {e}")
                # Wait an hour before retrying on error
                await asyncio.sleep(3600)
async def main():
    """Main function to run the scheduler."""
    try:
        # Get settings
        settings = get_settings()
        logger.info(f"Initializing consolidation scheduler with settings:")
        #These are now in config
        #logger.info(f"  Run time: {settings.consolidation_hour:02d}:{settings.consolidation_minute:02d}")
        #logger.info(f"  Timezone: {settings.timezone}")
        #logger.info(f"  Batch size: {settings.consolidation_batch_size}")

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
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())