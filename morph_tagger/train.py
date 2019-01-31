# Max sentence length
import os
import pickle

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders import ConllDataset
from data_utils import read_surfaces
from eval import read_conllu, manipulate_data, input_pairs
from layers import EncoderRNN, DecoderRNN
from logger import LOGGER


# Learning rate decay
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


def predict(surface_words, encoder, decoder_lemma, decoder_morph_tags, dataset, device=torch.device("cpu"),
            max_lemma_len=20, max_morph_features_len=10):
    """

    Args:
        surface_words (list): List of tokens (str)
        encoder (`layers.EncoderRNN`): Encoder RNN
        decoder_lemma (`layers.DecoderRNN`): Lemma Decoder
        decoder_morph_tags (`layers.DecoderRNN`): Morphological Features Decoder
        dataset (`torch.utils.data.Dataset`): Train Dataset. Required for vocab etc.
        device (`torch.device`): Default is cpu
        max_lemma_len (int): Maximum length of lemmas
        max_morph_features_len (int): Maximum length of morphological features

    Returns:
        str: Predicted conll sentence
    """

    if len(surface_words) == 0:
        return ""

    max_token_len = max([len(surface) for surface in surface_words])+1

    encoded_surfaces = torch.zeros((len(surface_words), max_token_len), dtype=torch.long)
    for ix, surface in enumerate(surface_words):
        encoded_surface = dataset.encode(surface, dataset.surface_char2id)
        encoded_surfaces[ix, :encoded_surface.size()[0]] = encoded_surface

    encoded_surfaces = encoded_surfaces.to(device)

    # Run encoder
    word_representations, context_aware_representations = encoder(encoded_surfaces.view(1, *encoded_surfaces.size()))

    # Run lemma decoder for each word
    lemmas = []
    words_count = context_aware_representations.size(0)
    for i in range(words_count):
        _, lemma = decoder_lemma.predict(word_representations[i], context_aware_representations[i],
                                         max_len=max_lemma_len, device=device)
        lemmas.append(''.join(lemma))

    # Run morph features decoder for each word
    morph_features = []
    for i in range(words_count):
        _, morph_feature = decoder_morph_tags.predict(word_representations[i], context_aware_representations[i],
                                                      max_len=max_morph_features_len, device=device)
        morph_features.append(';'.join(morph_feature))

    conll_sentence = "# Sentence\n"
    for i, (surface, lemma, morph_feature) in enumerate(zip(surface_words, lemmas, morph_features)):
        conll_sentence += "{}\t{}\t{}\t_\t_\t{}\t_\t_\t_\t_\n".format(i+1, surface, lemma, morph_feature)
    return conll_sentence


def train():
    # Select cuda as device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    LOGGER.info("Using {} as default device".format(device))

    max_words = 50

    # Number of epochs
    num_epochs = 100
    patience = 4

    # Encoder hyper-parmeters
    embedding_size = 64
    char_gru_hidden_size = 512
    word_gru_hidden_size = 512
    encoder_dropout = 0.2

    # Decoder hyper-parmeters
    output_embedding_size = 256
    decoder_dropout = 0.2

    data_path = '../data/2019/task2/'
    language_paths = [data_path + filename for filename in os.listdir(data_path)]
    language_names = [filename.replace('UD_', '') for filename in os.listdir(data_path)]

    # Iterate over languages
    for language_path, language_name in zip(language_paths, language_names):
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
        train_set = ConllDataset(train_data_path)
        train_loader = DataLoader(train_set)

        # Load validation data
        val_set = ConllDataset(val_data_path, surface_char2id=train_set.surface_char2id,
                               lemma_char2id=train_set.lemma_char2id, morph_tag2id=train_set.morph_tag2id, mode='test')
        val_loader = DataLoader(val_set)
        val_data_surface_words = read_surfaces(val_data_path)

        # Build Models
        # Initialize encoder and decoders
        LOGGER.info('Building models for language: {}'.format(language_name))
        encoder = EncoderRNN(embedding_size, char_gru_hidden_size, word_gru_hidden_size,
                             len(train_set.surface_char2id), dropout_ratio=encoder_dropout, device=device)
        encoder = encoder.to(device)

        decoder_lemma = DecoderRNN(output_embedding_size, word_gru_hidden_size, train_set.lemma_char2id,
                                   dropout_ratio=decoder_dropout).to(device)

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

        encoder_scheduler = MultiStepLR(encoder_optimizer, milestones=[5, 10, 20, 30, 50, 80],
                                        gamma=0.2)
        decoder_lemma_scheduler = MultiStepLR(decoder_lemma_optimizer,
                                              milestones=[5, 10, 20, 30, 50, 80],
                                              gamma=0.2)
        decoder_morph_tags_scheduler = MultiStepLR(decoder_morph_tags_optimizer,
                                                   milestones=[5, 10, 20, 30, 50, 80],
                                                   gamma=0.2)
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

            for x, y1, y2 in tqdm(train_loader, desc='Training'):
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

                # Run encoder
                word_embeddings, context_embeddings = encoder(x)

                # Run decoder for each word
                sentence_loss = 0.0

                for j, (_y, decoder) in enumerate(zip([y1, y2], [decoder_lemma, decoder_morph_tags])):
                    decoder_outputs = decoder(word_embeddings, context_embeddings, _y[0, :, :-1])
                    word_count = word_embeddings.size(0)
                    for word_ix in range(word_count):
                        sentence_loss += criterion(decoder_outputs[word_ix], _y[0, word_ix, 1:])

                    sentence_loss.backward(retain_graph=True)

                    # Optimization
                    encoder_optimizer.step()
                    decoder_lemma_optimizer.step()
                    decoder_morph_tags_optimizer.step()
                    total_train_loss += sentence_loss.item() / (word_count * 2.0)
                    if j == 0:
                        lemma_loss += sentence_loss.item() / (word_count * 1.0)
                    else:
                        morph_loss += sentence_loss.item() / (word_count * 1.0)

            encoder.eval()
            decoder_lemma.eval()
            decoder_morph_tags.eval()
            for x, y1, y2 in tqdm(val_loader, desc='Validation'):
                # Skip sentences longer than max_words
                if x.size(1) > max_words:
                    continue

                # Send input to the device
                x = x.to(device)
                y1 = y1.to(device)
                y2 = y2.to(device)

                # Run encoder
                word_embeddings, context_embeddings = encoder(x)

                # Run decoder for each word
                sentence_loss = 0.0

                for j, (_y, decoder) in enumerate(zip([y1, y2], [decoder_lemma, decoder_morph_tags])):
                    decoder_outputs = decoder(word_embeddings, context_embeddings, _y[0, :, :-1])
                    word_count = word_embeddings.size(0)
                    for word_ix in range(word_count):
                        sentence_loss += criterion(decoder_outputs[word_ix], _y[0, word_ix, 1:])

                    val_loss += sentence_loss.item() / (word_count * 2.0)
                    if j == 0:
                        val_lemma_loss += sentence_loss.item() / (word_count * 1.0)
                    else:
                        val_morph_loss += sentence_loss.item() / (word_count * 1.0)

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

        LOGGER.info('Prediction over validation data...')
        encoder.eval()
        decoder_lemma.eval()
        decoder_morph_tags.eval()
        # Make predictions and save to file
        prediction_file = train_data_path.replace('train', 'predictions')
        with open(prediction_file, 'w', encoding='UTF-8') as f:
            for sentence in val_data_surface_words:
                conll_sentence = predict(sentence, encoder, decoder_lemma, decoder_morph_tags,
                                         train_set, device=device)
                f.write(conll_sentence)
                f.write('\n')

        # Evaluate
        LOGGER.info('Evaluating...')
        reference = read_conllu(val_data_path)
        output = read_conllu(prediction_file)
        results = manipulate_data(input_pairs(reference, output))

        LOGGER.info('Evaluation completed')
        LOGGER.info('Lemma Acc:{}, Lemma Lev. Dist: {}, Morph acc: {}, F1: {} '.format(*results))

        LOGGER.info('Writing results to file...')
        # save results
        with open('results.txt', 'a', encoding='UTF-8') as f:
            f.write('Language: {}, Lemma Acc:{}, Lemma Lev. Dist: {}, Morph acc: {}, F1: {} \n'.format(
                language_name, *results
            ))

        # save models
        LOGGER.info('Saving models...')
        torch.save(encoder.state_dict(), train_data_path.replace('train', 'encoder').replace('conllu', 'model'))
        torch.save(decoder_lemma.state_dict(), train_data_path.replace('train', 'decoder_lemma').replace('conllu', 'model'))
        torch.save(decoder_morph_tags.state_dict(), train_data_path.replace('train', 'decoder_morph').replace('conllu', 'model'))
        with open(train_data_path.replace('-train', '').replace('conllu', 'dataset'), 'wb') as f:
            pickle.dump(train_set, f)

if __name__ == '__main__':
    train()
