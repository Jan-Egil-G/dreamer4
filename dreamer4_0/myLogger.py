import logging
import os

def setup_logger():
    # Create the directory if it doesn't exist
    os.makedirs('data/debuglog', exist_ok=True)
    
    # Clear any existing handlers
    logging.root.handlers.clear()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/debuglog/debug1.log', mode='w')
        ],
        force=True  # This forces reconfiguration
    )