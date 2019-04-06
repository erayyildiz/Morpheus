import os
import pickle

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders import ConllDataset
from eval import evaluate, evaluate_all
from languages import PILOT_LANGUAGES
from layers import EncoderRNN, DecoderRNN, TransformerRNN
from logger import LOGGER


# Encoder hyper-parmeters
embedding_size = 128
char_gru_hidden_size = 1024
word_gru_hidden_size = 1024
encoder_dropout = 0.5

# Decoder hyper-parmeters
output_embedding_size = 256
decoder_dropout = 0.5

only_pivot_languages = False
model_name = 'LemmaTransformer'

# Number of epochs
num_epochs = 100
max_sentences = 0
patience = 4

# Select cuda as device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
LOGGER.info("Using {} as default device".format(device))


def lr_decay_step(lr, model, factor=0.1, weight_decay=0.0):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return lr, optimizer


def train():
    max_words = 50

    data_path = '../data/2019/task2/'
    language_paths = [data_path + filename for filename in os.listdir(data_path)]
    language_names = [filename.replace('UD_', '') for filename in os.listdir(data_path)]

    # Iterate over languages
    for language_ix, (language_path, language_name) in enumerate(zip(language_paths, language_names)):
        if language_ix < 48:
            continue
        # Skip langauges not in pilot languages array (if only_pivot_languages is True)
        if only_pivot_languages and language_name not in PILOT_LANGUAGES:
            continue
        try:
            LOGGER.info('{}th LANGUAGE: {}'.format(language_ix, language_name))

            # Read dataset for language
            LOGGER.info('Reading files for language: {}'.format(language_name))
            language_conll_files = os.listdir(language_path)
            train_data_path = None
            val_data_path = None

            for language_conll_file in language_conll_files:

                if 'train.' in language_conll_file:
                    train_data_path = language_path + '/' + language_conll_file
                elif 'dev.' in language_conll_file:
                    val_data_path = language_path + '/' + language_conll_file

            assert train_data_path, 'Training data not found'
            assert val_data_path, 'Validation data not found'

            # Load train set
            train_set = ConllDataset(train_data_path, max_sentences=max_sentences)
            train_loader = DataLoader(train_set)

            # Load validation data
            val_set = ConllDataset(val_data_path, surface_char2id=train_set.surface_char2id,
                                   lemma_char2id=train_set.lemma_char2id, morph_tag2id=train_set.morph_tag2id,
                                   transformation2id=train_set.transformation2id,
                                   mode='test', max_sentences=max_sentences)
            val_loader = DataLoader(val_set)

            # Build Models
            # Initialize encoder and decoders
            LOGGER.info('Building models for language: {}'.format(language_name))
            encoder = EncoderRNN(embedding_size, char_gru_hidden_size, word_gru_hidden_size,
                                 len(train_set.surface_char2id), dropout_ratio=encoder_dropout, device=device)

            encoder = encoder.to(device)

            decoder_lemma = TransformerRNN(output_embedding_size, word_gru_hidden_size, train_set.transformation2id,
                                           len(train_set.surface_char2id), dropout_ratio=decoder_dropout).to(device)

            decoder_morph_tags = DecoderRNN(output_embedding_size, word_gru_hidden_size, train_set.morph_tag2id,
                                            dropout_ratio=decoder_dropout).to(device)

            # Define loss and optimizers
            criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

            # Create optimizers
            encoder_lr = 0.0003
            decoder_lemma_lr = 0.0003
            decoder_morph_lr = 0.0003
            encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
            decoder_lemma_optimizer = torch.optim.Adam(decoder_lemma.parameters(), lr=decoder_lemma_lr)
            decoder_morph_tags_optimizer = torch.optim.Adam(decoder_morph_tags.parameters(), lr=decoder_morph_lr)

            encoder_scheduler = MultiStepLR(encoder_optimizer, milestones=list(range(5, 100, 1)), gamma=0.5)
            decoder_lemma_scheduler = MultiStepLR(decoder_lemma_optimizer, milestones=list(range(5, 100, 1)), gamma=0.5)
            decoder_morph_tags_scheduler = MultiStepLR(decoder_morph_tags_optimizer, milestones=list(range(5, 100, 1)),
                                                       gamma=0.5)

            prev_val_loss = 1000000
            num_epochs_wo_improvement = 0

            LOGGER.info('Training starts for language: {}'.format(language_name))
            # Let the training begin
            for epoch in range(num_epochs):
                LOGGER.info('Epoch {} starts'.format(epoch))
                total_train_loss = 0.0
                lemma_loss = 0.0
                morph_loss = 0.0
                val_loss = 0.0
                val_lemma_loss = 0.0
                val_morph_loss = 0.0

                # Training part
                encoder.train()
                decoder_lemma.train()
                decoder_morph_tags.train()

                # LR Schedule
                encoder_scheduler.step()
                decoder_lemma_scheduler.step()
                decoder_morph_tags_scheduler.step()

                for x, y1, y2, y3 in tqdm(train_loader, desc='Training'):
                    # Skip sentences longer than max_words
                    if x.size(1) > max_words:
                        continue

                    # Clear gradients for each sentence
                    encoder.zero_grad()
                    decoder_lemma.zero_grad()
                    decoder_morph_tags.zero_grad()

                    # Send input to the device
                    x = x.to(device)
                    # y1 = y1.to(device)
                    y2 = y2.to(device)
                    y3 = y3.to(device)

                    # Run encoder
                    word_embeddings, context_embeddings = encoder(x)

                    # Run morph decoder for each word
                    sentence_loss = 0.0
                    morph_decoder_outputs = decoder_morph_tags(word_embeddings, context_embeddings, y2[0, :, :-1])
                    word_count = word_embeddings.size(0)
                    for word_ix in range(word_count):
                        sentence_loss += criterion(morph_decoder_outputs[word_ix], y2[0, word_ix, 1:])

                    sentence_loss.backward(retain_graph=True)
                    total_train_loss += sentence_loss.item() / (word_count * 2.0)
                    morph_loss += sentence_loss.item() / (word_count * 1.0)

                    lemma_decoder_outputs = decoder_lemma(word_embeddings, context_embeddings, x)
                    word_count = word_embeddings.size(0)
                    for word_ix in range(word_count):
                        sentence_loss += criterion(lemma_decoder_outputs[word_ix], y3[0, word_ix, :])

                    sentence_loss.backward(retain_graph=True)
                    total_train_loss += sentence_loss.item() / (word_count * 2.0)
                    lemma_loss += sentence_loss.item() / (word_count * 1.0)

                    encoder_optimizer.step()
                    decoder_lemma_optimizer.step()
                    decoder_morph_tags_optimizer.step()

                encoder.eval()
                decoder_lemma.eval()
                decoder_morph_tags.eval()
                for x, y1, y2, y3 in tqdm(val_loader, desc='Validation'):
                    # Skip sentences longer than max_words
                    if x.size(1) > max_words:
                        continue

                    # Send input to the device
                    x = x.to(device)
                    # y1 = y1.to(device)
                    y2 = y2.to(device)
                    y3 = y3.to(device)

                    # Run encoder
                    word_embeddings, context_embeddings = encoder(x)

                    # Run morph decoder
                    sentence_loss = 0.0
                    morph_decoder_outputs = decoder_morph_tags(word_embeddings, context_embeddings, y2[0, :, :-1])
                    word_count = word_embeddings.size(0)
                    for word_ix in range(word_count):
                        sentence_loss += criterion(morph_decoder_outputs[word_ix], y2[0, word_ix, 1:])

                    val_loss += sentence_loss.item() / (word_count * 2.0)
                    val_morph_loss += sentence_loss.item() / (word_count * 1.0)

                    # Run lemma decoder
                    sentence_loss = 0.0
                    lemma_decoder_outputs = decoder_lemma(word_embeddings, context_embeddings, x)
                    word_count = word_embeddings.size(0)
                    for word_ix in range(word_count):
                        sentence_loss += criterion(lemma_decoder_outputs[word_ix], y3[0, word_ix, :])

                    val_loss += sentence_loss.item() / (word_count * 2.0)
                    val_lemma_loss += sentence_loss.item() / (word_count * 1.0)

                LOGGER.info('Epoch {0:3d}, Loss: {1:7.3f}, Lemma Loss: {2:7.3f}, Morph Loss: {3:7.3f}'.format(
                    epoch,
                    (1.0 * total_train_loss) / len(train_loader),
                    (1.0 * lemma_loss) / len(train_loader),
                    (1.0 * morph_loss) / len(train_loader)
                ))
                LOGGER.info('Val Loss: {1:7.3f}, Val Lemma Loss: {2:7.3f}, Val Morph Loss: {3:7.3f}'.format(
                    epoch,
                    (1.0 * val_loss) / len(val_loader),
                    (1.0 * val_lemma_loss) / len(val_loader),
                    (1.0 * val_morph_loss) / len(val_loader)
                ))

                if val_loss >= prev_val_loss:
                    num_epochs_wo_improvement += 1
                    if num_epochs_wo_improvement >= patience:
                        break
                else:
                    num_epochs_wo_improvement = 0
                    prev_val_loss = val_loss

                    # save models
                    LOGGER.info('Acc Increased, Saving models...')
                    torch.save(encoder.state_dict(),
                               train_data_path.replace('train', 'encoder').replace('conllu', '{}.model'.format(model_name)))
                    torch.save(decoder_lemma.state_dict(),
                               train_data_path.replace('train', 'decoder_lemma').replace('conllu',
                                                                                         '{}.model'.format(model_name)))
                    torch.save(decoder_morph_tags.state_dict(),
                               train_data_path.replace('train', 'decoder_morph').replace('conllu',
                                                                                         '{}.model'.format(model_name)))
                    with open(train_data_path.replace('-train', '').replace('conllu', '{}.dataset'.format(model_name)),
                              'wb') as f:
                        pickle.dump(train_set, f)

            # Make predictions and save to file
            LOGGER.info('Training completed')
            LOGGER.info('Evaluation...')
            from predict import predict
            predict(language_path, model_name, val_data_path)
            eval_results = evaluate(language_name, language_path, model_name=model_name)

            LOGGER.info('Evaluation completed')
            for k, v in eval_results.items():
                print('{}: {}'.format(k, v))
        except Exception as e:
            LOGGER.error(e)

    # evaluate_all(model_name=model_name)

if __name__ == '__main__':
    train()
    # evaluate_all(model_name=model_name)
