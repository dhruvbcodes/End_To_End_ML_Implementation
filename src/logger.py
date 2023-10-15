import logging
import os
from datetime import datetime

Log_file = f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), 'logs', 'Log_file')
os.makedirs(logs_path, exist_ok=True)      

Log_file_path = os.path.join(logs_path, Log_file)

logging.basicConfig(
    filename=Log_file_path, 
    level=logging.INFO,
    format='%(asctime)s : %(lineno)d %(name)s : %(levelname)s : %(message)s'
)

'''
#to check if logger works lmao
if __name__ == "__main__":
    logging.info("Logging has started")
'''
