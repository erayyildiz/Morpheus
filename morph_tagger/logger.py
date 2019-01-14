import logging
import sys


LOG_FORMAT = '%(asctime)s, %(levelname)-8s %(message)s'

LOGGER = logging.getLogger('MorphTagger')
logging.basicConfig(format=LOG_FORMAT, datefmt='%d-%m-%Y:%H:%M:%S', level=logging.INFO, stream=sys.stdout)
