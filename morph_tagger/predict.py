import pickle

import torch
import os

from data_utils import read_surfaces
from layers import EncoderRNN, DecoderRNN
from logger import LOGGER

# Encoder hyper-parmeters
embedding_size = 64
char_gru_hidden_size = 512
word_gru_hidden_size = 512
encoder_dropout = 0.2

# Decoder hyper-parmeters
output_embedding_size = 256
decoder_dropout = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_sentence(surface_words, encoder, decoder_lemma, decoder_morph_tags, dataset, device=torch.device("cpu"),
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


def predict(language_path, model_name, conll_file):
    language_conll_files = os.listdir(language_path)
    for language_conll_file in language_conll_files:
        if 'train.' in language_conll_file:

            # LOAD DATASET
            LOGGER.info('Loading dataset...')
            train_data_path = language_path + '/' + language_conll_file
            train_set = None
            with open(train_data_path.replace('-train', '').replace('conllu', '{}.dataset'.format(model_name)), 'rb') as f:
                train_set = pickle.load(f)
            data_surface_words = read_surfaces(language_path + '/' + conll_file)

            # LOAD ENCODER MODEL
            LOGGER.info('Loading Encoder...')
            encoder = EncoderRNN(embedding_size, char_gru_hidden_size, word_gru_hidden_size,
                                 len(train_set.surface_char2id), dropout_ratio=encoder_dropout, device=device)
            encoder.load_state_dict(torch.load(
                train_data_path.replace('train', 'encoder').replace('conllu', '{}.model'.format(model_name))
            ))
            encoder = encoder.to(device)

            # LOAD LEMMA DECODER MODEL
            LOGGER.info('Loading Lemma Decoder...')
            decoder_lemma = DecoderRNN(output_embedding_size, word_gru_hidden_size, train_set.lemma_char2id,
                                       dropout_ratio=decoder_dropout).to(device)
            decoder_lemma.load_state_dict(torch.load(
                train_data_path.replace('train', 'decoder_lemma').replace('conllu', '{}.model'.format(model_name))
            ))
            decoder_lemma = decoder_lemma.to(device)

            # LOAD MORPH DECODER MODEL
            LOGGER.info('Loading Morph Decoder...')
            decoder_morph_tags = DecoderRNN(output_embedding_size, word_gru_hidden_size, train_set.lemma_char2id,
                                            dropout_ratio=decoder_dropout).to(device)
            decoder_morph_tags.load_state_dict(torch.load(
                train_data_path.replace('train', 'decoder_morph').replace('conllu', '{}.model'.format(model_name))
            ))
            decoder_morph_tags = decoder_morph_tags.to(device)

            encoder.eval()
            decoder_lemma.eval()
            decoder_morph_tags.eval()

            # Make predictions and save to file
            prediction_file = train_data_path.replace('train', 'predictions-{}'.format(model_name))
            with open(prediction_file, 'w', encoding='UTF-8') as f:
                for sentence in data_surface_words:
                    conll_sentence = predict_sentence(sentence, encoder, decoder_lemma, decoder_morph_tags,
                                                      train_set, device=device)
                    f.write(conll_sentence)
                    f.write('\n')


if __name__ == '__main__':
    predict('../data/2019/task2/UD_Turkish-PUD', 'standard_morphnet', 'tr_pud-um-dev.conllu')