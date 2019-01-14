# Max sentence length
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loaders import ConllDataset
from layers import EncoderRNN, DecoderRNN
from logger import LOGGER


# Select cuda as device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info("Using {} as default device".format(device))

max_words = 50

# Sentences are batches, do not change
BATCH_SIZE = 1

# Number of epochs
num_epochs = 500
notify_each = 50
update_lr = 1000

# Encoder hyper-parmeters
embedding_size = 64
char_gru_hidden_size = 512
word_gru_hidden_size = 512
encoder_lr = 0.01
encoder_dropout = 0.3
encoder_weight_decay = 0.01
encoder_scheduler_factor = 0.9
encoder_momentum = 0.1

# Decoder hyper-parmeters
output_embedding_size = 256
decoder_gru_hidden_size = 512
decoder_lr = 0.01
decoder_dropout = 0.3
decoder_weight_decay = 0.01
decoder_scheduler_factor = 0.9
decoder_momentum = 0.1

data_path = '../data/2019/task2/'
language_paths = [data_path + filename for filename in os.listdir(data_path)]
language_names = [filename.replace('UD_', '') for filename in os.listdir(data_path)]


# Learning rate decay
def lr_decay_step(lr, factor=0.1, weight_decay=0.0):
    """Learning rate decay step

    Args:
        lr:
        factor:
        weight_decay:
    Returns:
        int: lr
        optimizer
    """
    lr *= factor
    optimizer = torch.optim.SGD(encoder.parameters(),
                                lr=lr, weight_decay=weight_decay
                                )
    return lr, optimizer


# Iterate over languages
for language_path, language_name in zip(language_paths, language_names):
    # Read dataset for language
    LOGGER.info('Reading files for language: {}'.format(language_name))
    language_conll_files = os.listdir(language_path)
    assert len(language_conll_files) == 2, 'More than 2 files'
    for language_conll_file in language_conll_files:

        if 'train' in language_conll_file:
            train_data_path = language_path + '/' + language_conll_file
        else:
            val_data_path = language_path + '/' + language_conll_file

    # Load train set
    train_set = ConllDataset(train_data_path)
    # Load validation set
    val_set = ConllDataset(val_data_path,
                           surface_char2id=train_set.surface_char2id,
                           lemma_char2id=train_set.lemma_char2id,
                           morph_tag2id=train_set.morph_tag2id,
                           mode='test')

    train_loader = DataLoader(train_set)
    val_loader = DataLoader(val_set)

    # Build Models
    # Initialize encoder and decoders
    LOGGER.info('Building models for language: {}'.format(language_name))
    encoder = EncoderRNN(embedding_size, char_gru_hidden_size, word_gru_hidden_size,
                         len(train_set.surface_char2id), dropout_ratio=encoder_dropout)
    encoder = encoder.to(device)

    decoder_lemma = DecoderRNN(output_embedding_size, word_gru_hidden_size, train_set.lemma_char2id,
                               dropout_ratio=decoder_dropout).to(device)

    decoder_morph_tags = DecoderRNN(output_embedding_size, word_gru_hidden_size, train_set.lemma_char2id,
                                    dropout_ratio=decoder_dropout).to(device)

    # Define loss and optimizers
    criterion = nn.NLLLoss(ignore_index=0).to(device)

    # Create optimizers
    encoder_lr = 0.5
    decoder_lemma_lr = 0.5
    decoder_morph_tags_lr = 0.5
    encoder_optimizer = torch.optim.Adam(encoder.parameters(),
                                         lr=encoder_lr
                                         )
    decoder_lemma_optimizer = torch.optim.Adam(decoder_lemma.parameters(),
                                               lr=decoder_lemma_lr
                                               )

    decoder_morph_tags_optimizer = torch.optim.Adam(decoder_morph_tags.parameters(),
                                                    lr=decoder_morph_tags_lr
                                                    )

    # Let the training begin
    for epoch in range(num_epochs):

        total_train_loss = 0.0
        total_val_loss = 0.0
        previous_loss = 0.0
        # Training part
        encoder.train()
        decoder_lemma.train()
        decoder_morph_tags.train()

        for ix, (x, y1, y2) in enumerate(train_loader):
            # Skip sentences longer than max_words
            if x.size(1) > max_words:
                continue

            # Clear gradients for each sentence
            encoder.zero_grad()
            decoder_lemma.zero_grad()
            decoder_morph_tags.zero_grad()

            # Send input to the device
            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            # Init loss value
            loss = 0

            # Run encoder
            word_embeddings, context_embeddings = encoder(x)

            # Run decoder for each word
            word_count = context_embeddings.size(1)
            sentence_loss = 0.0
            for word_ix in range(word_count):
                for _y, decoder in zip([y1, y2], [decoder_lemma, decoder_morph_tags]):
                    target_length = _y[0][word_ix].size(0)
                    decoder_outputs = decoder(word_embeddings[0, word_ix], context_embeddings[0, word_ix],
                                              _y[0][word_ix], target_length=target_length)
                    loss += criterion(decoder_outputs, _y[0][word_ix])
                    sentence_loss += loss.item() / _y[0][word_ix].size(0)
            sentence_loss /= word_count
            total_train_loss += sentence_loss

            # Optimization
            loss.backward()

            encoder_optimizer.step()
            decoder_lemma_optimizer.step()
            decoder_morph_tags_optimizer.step()

            if (ix + 1) % notify_each == 0:
                print("Epoch {}. Sample: {}. Train Loss: {}, Encoder lr: {}, Decoder lr: {}".format(
                    epoch, (ix + 1), total_train_loss / (ix + 1), encoder_lr, decoder_lr)
                )
