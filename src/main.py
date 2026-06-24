import sys
import logging
import uvicorn
from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Bootstrap")

def main():
    """
    Entry Point.
    Its sole responsibility is to route the execution based on the config.
    """
    mode = settings.get("execution_mode", "cli").lower()

    if mode == "cli":
        logger.info("Starting in Terminal Mode (CLI)...")
        from src.cli.terminal import run_cli
        run_cli()

    elif mode == "api":
        logger.info("Starting Web Server (FastAPI)...")
        # Launch Uvicorn pointing to the app in src.api.app
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

    else:
        logger.error(f"❌ Invalid mode '{mode}'. Please check settings.yaml.")
        sys.exit(1)

if __name__ == "__main__":
    main()