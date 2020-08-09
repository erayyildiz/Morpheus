import os
import pickle

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import optparse

from configs import *
from data_loaders import ConllDataset
from eval import evaluate
from languages import PILOT_LANGUAGES, NON_TRANSFORMER_LANGUAGES
from layers import EncoderRNN, DecoderRNN, TransformerRNN, DecoderFF
from logger import LOGGER

# Encoder hyper-parmeters
embedding_size = CHAR_EMBEDDING_SIZE
char_gru_hidden_size = CHAR_GRU_HIDDEN_SIZE
word_gru_hidden_size = WORD_GRU_HIDDEN_SIZE
encoder_dropout = ENCODE_DROPOUT

# Decoder hyper-parmeters
output_embedding_size = OUTPUT_EMBEDDING_SIZE
decoder_dropout = DECODER_DROPOUT

# Select cuda as device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info("Using {} as default device".format(device))


def train(language_name, train_data_path, val_data_path, use_min_edit_operation_decoder=True,
          encoder_lr=0.0003, decoder_lemma_lr=0.00003, decoder_morph_lr=0.0003, transformer_lr=0.0003, max_words=50,
          use_transformer=True, use_char_lstm=True, use_rnn_morph=False, transformer_model_name=TRANSFORMER_MODEL_NAME,
          model_name='LemmaTransformer', patience=4, num_epochs=100):
    """
    Trains model for given language.

    Args:

        language_name (str): Name of the language
        train_data_path (str): The conll file path for training
        val_data_path (str): The conll file path for validation
        use_min_edit_operation_decoder (bool): USe minimum edit operation decoder instead of charecter prediction decoder
            Default is True.
        encoder_lr (float): Learning rate for encoder
        decoder_lemma_lr (float): Learning rate for lemma decoder
        decoder_morph_lr (float): Learning rate for morphological tagging decoder
        max_words (int): Maximum number of words in a sentence
        use_transformer (str): indicates weather to use transformer
        use_char_lstm (str): indicate whether to use char lstm
        use_rnn_morph (str): indicate whether to use lstm to dcode morph tags
        transformer_model_name (str): HuggingFace style transformer name
        model_name (str): Name of the model
        patience (int): Number of epochs without improvement. Used for early stopping
        num_epochs (int): Maximum number of epochs. Default is 100.


    """

    assert use_char_lstm or use_transformer, 'One of use_char_lstm and use_transformer should be True'
    if use_transformer:
        assert transformer_model_name, 'transformer_model_name should be provided if use_transformer is True'

    # Load train set
    train_set = ConllDataset(train_data_path, transformer_model_name=transformer_model_name)
    train_loader = DataLoader(train_set)

    # Load validation data
    val_set = ConllDataset(val_data_path, surface_char2id=train_set.surface_char2id,
                           lemma_char2id=train_set.lemma_char2id, morph_tag2id=train_set.morph_tag2id,
                           transformer_model_name=transformer_model_name,
                           transformation2id=train_set.transformation2id, mode='test')
    val_loader = DataLoader(val_set)

    # Build Models
    # Initialize encoder and decoders
    LOGGER.info('Building models for language: {}'.format(language_name))
    if use_transformer:
        encoder = EncoderRNN(embedding_size, char_gru_hidden_size, word_gru_hidden_size,
                             len(train_set.surface_char2id), TRANSFORMER_MODEL_NAME,
                             dropout_ratio=encoder_dropout, device=device)
    else:
        encoder = EncoderRNN(embedding_size, char_gru_hidden_size, word_gru_hidden_size,
                             len(train_set.surface_char2id), None,
                             dropout_ratio=encoder_dropout, device=device)

    encoder = encoder.to(device)

    if language_name in NON_TRANSFORMER_LANGUAGES or not use_min_edit_operation_decoder:
        if use_transformer:
            decoder_lemma = DecoderRNN(output_embedding_size, word_gru_hidden_size, train_set.lemma_char2id,
                                       layer_size=3, dropout_ratio=decoder_dropout).to(device)
        else:
            decoder_lemma = DecoderRNN(output_embedding_size, word_gru_hidden_size, train_set.lemma_char2id,
                                       layer_size=2, dropout_ratio=decoder_dropout).to(device)
    else:
        if use_transformer:
            decoder_lemma = TransformerRNN(output_embedding_size, word_gru_hidden_size, train_set.transformation2id,
                                           len(train_set.surface_char2id), layer_size=3,
                                           dropout_ratio=decoder_dropout).to(device)
        else:
            decoder_lemma = TransformerRNN(output_embedding_size, word_gru_hidden_size, train_set.transformation2id,
                                           len(train_set.surface_char2id), layer_size=3,
                                           dropout_ratio=decoder_dropout).to(device)

    if use_transformer and use_rnn_morph:
        decoder_morph_tags = DecoderRNN(output_embedding_size, word_gru_hidden_size, train_set.morph_tag2id,
                                        layer_size=3, dropout_ratio=decoder_dropout).to(device)
    elif use_rnn_morph:
        decoder_morph_tags = DecoderRNN(output_embedding_size, word_gru_hidden_size, train_set.morph_tag2id,
                                        layer_size=2, dropout_ratio=decoder_dropout).to(device)
    else:
        decoder_morph_tags = DecoderFF(word_gru_hidden_size, train_set.morph_tag2id,
                                       dropout_ratio=decoder_dropout).to(device)

    # Define loss and optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    bce_criterion = nn.BCEWithLogitsLoss().to(device)

    # Create optimizers
    parameters_except_based_model = [x[1] for x in encoder.named_parameters() if 'based_model' not in x[0]]
    encoder_optimizer = torch.optim.Adam([
        {"params": parameters_except_based_model, "lr": encoder_lr}
    ], lr=encoder_lr)
    transformer_optimizer = torch.optim.Adam(
            [
                {"params": encoder.based_model.parameters(), "lr": transformer_lr},
            ],
            lr=transformer_lr, weight_decay=0.01
        )
    decoder_lemma_optimizer = torch.optim.Adam(decoder_lemma.parameters(), lr=decoder_lemma_lr)
    decoder_morph_tags_optimizer = torch.optim.Adam(decoder_morph_tags.parameters(), lr=decoder_morph_lr)

    encoder_scheduler = MultiStepLR(encoder_optimizer, milestones=list(range(3, 100, 2)), gamma=0.8)
    transformer_scheduler = MultiStepLR(transformer_optimizer, milestones=list(range(3, 100, 2)), gamma=0.8)
    decoder_lemma_scheduler = MultiStepLR(decoder_lemma_optimizer, milestones=list(range(3, 100, 2)), gamma=0.8)
    decoder_morph_tags_scheduler = MultiStepLR(decoder_morph_tags_optimizer, milestones=list(range(3, 100, 2)),
                                               gamma=0.8)

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

        for x1, x2, y1, y2, y3 in tqdm(train_loader, desc='Training'):
            # Skip sentences longer than max_words
            if x2.size(1) > max_words:
                continue

            # Clear gradients for each sentence
            encoder.zero_grad()
            decoder_lemma.zero_grad()
            decoder_morph_tags.zero_grad()

            # Send input to the device
            if use_transformer:
                x1 = (x1[0].to(device), x1[1].to(device))
            x2 = x2.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)

            # Run encoder
            transformer_output, word_embeddings, context_embeddings = encoder(x1, x2)

            # Run morph decoder for each word
            sentence_loss = 0.0
            if use_transformer and use_rnn_morph:
                morph_decoder_outputs = decoder_morph_tags(word_embeddings, context_embeddings,
                                                           y2[0, :, :-1], transformer_context=transformer_output)
            elif use_transformer:
                morph_decoder_outputs = decoder_morph_tags(word_embeddings, context_embeddings,
                                                           transformer_context=transformer_output)
            elif use_rnn_morph:
                morph_decoder_outputs = decoder_morph_tags(word_embeddings, context_embeddings)
            else:
                morph_decoder_outputs = decoder_morph_tags(word_embeddings, context_embeddings, y2[0, :, :-1])

            word_count = word_embeddings.size(0)
            if use_rnn_morph:
                for word_ix in range(word_count):
                    sentence_loss += criterion(morph_decoder_outputs[word_ix], y2[0, word_ix, 1:])
            else:
                y_onehot = torch.FloatTensor(*morph_decoder_outputs.size()).to(device)
                y_onehot.zero_()
                y_onehot.scatter_(1, y2[0], 1)
                sentence_loss += bce_criterion(morph_decoder_outputs, y_onehot)

            sentence_loss.backward(retain_graph=True)
            total_train_loss += sentence_loss.item() / (word_count * 2.0)
            morph_loss += sentence_loss.item() / (word_count * 1.0)

            if isinstance(decoder_lemma, TransformerRNN):
                if use_transformer:
                    lemma_decoder_outputs = decoder_lemma(word_embeddings, context_embeddings, x2,
                                                          transformer_context=transformer_output)
                else:
                    lemma_decoder_outputs = decoder_lemma(word_embeddings, context_embeddings, x2)
            else:
                if use_transformer:
                    lemma_decoder_outputs = decoder_lemma(word_embeddings, context_embeddings, y1[0, :, :-1],
                                                          transformer_context=transformer_output)
                else:
                    lemma_decoder_outputs = decoder_lemma(word_embeddings, context_embeddings, y1[0, :, :-1])

            word_count = word_embeddings.size(0)
            for word_ix in range(word_count):
                if isinstance(decoder_lemma, TransformerRNN):
                    sentence_loss += criterion(lemma_decoder_outputs[word_ix], y3[0, word_ix, :])
                else:
                    sentence_loss += criterion(lemma_decoder_outputs[word_ix], y1[0, word_ix, 1:])

            sentence_loss.backward(retain_graph=True)
            total_train_loss += sentence_loss.item() / (word_count * 2.0)
            lemma_loss += sentence_loss.item() / (word_count * 1.0)

            encoder_optimizer.step()
            transformer_optimizer.step()
            decoder_lemma_optimizer.step()
            decoder_morph_tags_optimizer.step()

        encoder.eval()
        decoder_lemma.eval()
        decoder_morph_tags.eval()
        for x1, x2, y1, y2, y3 in tqdm(val_loader, desc='Validation'):
            # Skip sentences longer than max_words
            if x2.size(1) > max_words:
                continue

            # Send input to the device
            if use_transformer:
                x1 = (x1[0].to(device), x1[1].to(device))
            x2 = x2.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)

            # Run encoder
            transformer_output, word_embeddings, context_embeddings = encoder(x1, x2)

            # Run morph decoder for each word
            sentence_loss = 0.0
            if use_transformer and use_rnn_morph:
                morph_decoder_outputs = decoder_morph_tags(word_embeddings, context_embeddings,
                                                           y2[0, :, :-1], transformer_context=transformer_output)
            elif use_transformer:
                morph_decoder_outputs = decoder_morph_tags(word_embeddings, context_embeddings,
                                                           transformer_context=transformer_output)
            elif use_rnn_morph:
                morph_decoder_outputs = decoder_morph_tags(word_embeddings, context_embeddings)
            else:
                morph_decoder_outputs = decoder_morph_tags(word_embeddings, context_embeddings, y2[0, :, :-1])

            word_count = word_embeddings.size(0)
            if use_rnn_morph:
                for word_ix in range(word_count):
                    sentence_loss += criterion(morph_decoder_outputs[word_ix], y2[0, word_ix, 1:])
            else:
                y_onehot = torch.FloatTensor(*morph_decoder_outputs.size()).to(device)
                y_onehot.zero_()
                y_onehot.scatter_(1, y2[0], 1)
                sentence_loss += bce_criterion(morph_decoder_outputs, y_onehot)

            val_loss += sentence_loss.item() / (word_count * 2.0)
            val_morph_loss += sentence_loss.item() / (word_count * 1.0)

            # Run lemma decoder
            sentence_loss = 0.0
            if isinstance(decoder_lemma, TransformerRNN):
                if use_transformer:
                    lemma_decoder_outputs = decoder_lemma(word_embeddings, context_embeddings, x2,
                                                          transformer_context=transformer_output)
                else:
                    lemma_decoder_outputs = decoder_lemma(word_embeddings, context_embeddings, x2)
            else:
                if use_transformer:
                    lemma_decoder_outputs = decoder_lemma(word_embeddings, context_embeddings, y1[0, :, :-1],
                                                          transformer_context=transformer_output)
                else:
                    lemma_decoder_outputs = decoder_lemma(word_embeddings, context_embeddings, y1[0, :, :-1])
            word_count = word_embeddings.size(0)
            for word_ix in range(word_count):
                if isinstance(decoder_lemma, TransformerRNN):
                    sentence_loss += criterion(lemma_decoder_outputs[word_ix], y3[0, word_ix, :])
                else:
                    sentence_loss += criterion(lemma_decoder_outputs[word_ix], y1[0, word_ix, 1:])

            val_loss += sentence_loss.item() / (word_count * 2.0)
            val_lemma_loss += sentence_loss.item() / (word_count * 1.0)

        # LR Schedule
        encoder_scheduler.step()
        if epoch > 3:
            transformer_scheduler.step()
        decoder_lemma_scheduler.step()
        decoder_morph_tags_scheduler.step()

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
    from predict import predict_unimorph
    predict_unimorph(os.path.dirname(val_data_path), model_name, val_data_path,
                     use_min_edit_operation_decoder=use_min_edit_operation_decoder)
    eval_results = evaluate(language_name, os.path.dirname(val_data_path), model_name=model_name)

    LOGGER.info('Evaluation completed')
    for k, v in eval_results.items():
        print('{}: {}'.format(k, v))


def train_all(data_path='../data/2019/task2/', only_pivot_languages=True, **params):
    """
    Trains models for all languages in a given data folder.

    Args:
        data_path (str): The path of the data folder where connl files located.
            The path should includes a folder for each language. See UniMorph dataset for details.
        only_pivot_languages: If True the training will be performed for only pivot languages that are
            expressed in `languages.py` file

    """

    language_paths = [data_path + filename for filename in os.listdir(data_path)]
    language_names = [filename.replace('UD_', '') for filename in os.listdir(data_path)]

    # Iterate over languages
    for language_ix, (language_path, language_name) in enumerate(zip(language_paths, language_names)):
        # Skip langauges not in pilot languages array (if only_pivot_languages is True)
        if only_pivot_languages and language_name not in PILOT_LANGUAGES:
            continue

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
        train(language_name, train_data_path, val_data_path, model_name='ElectraTransformer', **params)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.set_description('Train joint contextual lemmatizer and morphological tagger for given languages')

    parser.add_option('--all', '--all',
                      action="store_true", dest='train_all_languages',
                      help='use if you want to train models for all languages in UniMorph dataset', default=False)

    parser.add_option('-l', '--language_name',
                      action='store', dest='language_name',
                      help='The name of the language',
                      # default='Turkish-IMST'
                      )

    parser.add_option('-t', '--train_file',
                      action='store', dest='train_file',
                      help='CONLL file path for training',
                      # default='../data/2019/task2/UD_Turkish-IMST/tr_imst-um-train.conllu'
                      )

    parser.add_option('-d', '--dev_file',
                      action='store', dest='dev_file',
                      help='CONLL file path for validation',
                      # default='../data/2019/task2/UD_Turkish-IMST/tr_imst-um-dev.conllu'
                      )

    parser.add_option('-m', '--model_name',
                      action='store', dest='model_name',
                      help='Name for the model',
                      default='LemmaTransformer')

    options, args = parser.parse_args()

    if options.train_all_languages:
        train_all()
    elif all([options.language_name, options.train_file, options.dev_file, options.model_name]):
        train(language_name=options.language_name,
              train_data_path=options.train_file,
              val_data_path=options.dev_file,
              model_name=options.model_name)
    else:
        parser.print_help()
