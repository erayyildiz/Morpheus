
# Encoder hyper-parmeters
import torch

from logger import LOGGER

CHAR_EMBEDDING_SIZE = 128
CHAR_GRU_HIDDEN_SIZE = 1024
WORD_GRU_HIDDEN_SIZE = 1024
ENCODE_DROPOUT = 0.5
USE_TRANSFORMER = True
TRANSFORMER_MODEL_NAME = 'erayyildiz/electra-turkish-cased'
NUM_STEPS_BEFORE_BASE_MODEL_UPDATE = 3

# Decoder hyper-parmeters
OUTPUT_EMBEDDING_SIZE = 256
DECODER_DROPOUT = 0.5

# SELECT GPU IF AVAILABLE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info("Using {} as default device".format(DEVICE))