import logging

formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d:%(module)s:%(funcName)s] - %(message)s'
)

file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.addHandler(file_handler)
logger.addHandler(console_handler)