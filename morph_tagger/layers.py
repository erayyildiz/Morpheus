import torch
import torch.nn as nn
from tqdm import tqdm


class EncoderRNN(nn.Module):
    """A bidirectional GRU as the context encoder.

    The inputs are encoded words in a sentence.
    EncoderRNN firstly apply bidirectional GRU to characters of words to build word representations
    Then apply a second level bidirectional GRU to the word representations.
    By doing so it generates context-aware representations of each word in a sentence which then be used in decoding.

    """

    def __init__(self, embedding_size, hidden_size1, hidden_size2, vocab_len,
                 dropout_ratio=0.2, device=torch.device('cpu')):
        """ Initialize an EncoderRNN object

        Args:
            embedding_size (int): the dimension of the input character embeddings
            hidden_size1 (int): The number of units in first-level gru (char-gru)
            hidden_size2 (int): The number of units in second-level gru (context-gru)
            vocab_len (int): Number of unique characters to initialize character embedding module
            dropout_ratio(float): Dropout ratio, dropout applied to the outputs of both gru and embedding modules
        """
        super(EncoderRNN, self).__init__()

        # Hyper-parameters
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.embedding_size = embedding_size

        # Initialize modules
        self.embedding = nn.Embedding(vocab_len+1, embedding_size)
        self.char_gru = nn.GRU(embedding_size, hidden_size1, bidirectional=False, num_layers=1, batch_first=True)
        self.word_gru = nn.GRU(hidden_size1, hidden_size2, bidirectional=True, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_ratio)

        # Initialize hidden units
        self.char_gru_hidden = None
        self.word_gru_hidden = None

        self.device = device

    def init_context_hidden(self):
        """Initializes the hidden units of each context gru

        """
        return torch.zeros(2, 1, self.hidden_size2).to(self.device)

    def init_char_hidden(self, batch_size):
        """Initializes the hidden units of each char gru

        """
        return torch.zeros(1, batch_size, self.hidden_size1).to(self.device)

    def forward(self, x):
        """Forward pass of EncoderRNN

        Embedding layer, first level grus and second level grus are applied to input  tensor
        Dropouts are applied between all layers with parameters

        Returns:
            torch.Tensor: word embeddings generated by first-level character grus
            torch.Tensor: context-aware representations of the words occur in the sentence

        """
        # Batch size should be 1, sentences are batche in our implementation
        assert x.size(0) == 1, "Batch size should be 1 since each sentence is considered as a mini-batch"

        self.char_gru_hidden = self.init_char_hidden(x.size(1))
        self.word_gru_hidden = self.init_context_hidden()

        # Embedding layer
        char_embeddings = self.embedding(x)
        char_embeddings = self.dropout(char_embeddings)

        # First-level gru layer (char-gru to generate word embeddings)
        _, word_embeddings = self.char_gru(char_embeddings.view(char_embeddings.shape[1:]), self.char_gru_hidden)

        # Second-level gru layer (context-gru)
        context_embeddings = self.word_gru(word_embeddings, self.word_gru_hidden)[0]
        return word_embeddings[0], context_embeddings[0]


class DecoderRNN(nn.Module):
    """ The module generates characters and tags sequentially to construct a morphological analysis

    Inputs a context representation of a word and apply grus
    to predict the characters in the root form and the tags in the analysis respectively

    """

    def __init__(self, embedding_size, hidden_size, vocab, dropout_ratio=0):
        """Initialize the decoder object

        Args:
            embedding_size (int): The dimension of embeddings
                (output embeddings includes character for roots and tags for analyzes)
            hidden_size (int): The number of units in gru
            vocab (dict): Vocab dictionary where keys are either characters or tags and the values are integer
            dropout_ratio(float): Dropout ratio, dropout applied to the outputs of both gru and embedding modules
        """
        super(DecoderRNN, self).__init__()

        # Hyper parameters
        self.hidden_size = hidden_size

        # Vocab and inverse vocab to converts output indexes to characters and tags
        self.vocab = vocab
        self.index2token = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)

        # Layers
        self.W = nn.Linear(2 * hidden_size, hidden_size)
        self.embedding = nn.Embedding(len(vocab)+1, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, 2, batch_first=True)
        self.classifier = nn.Linear(hidden_size, len(vocab))
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, word_embeddings, context_vectors, y):
        """Forward pass of DecoderRNN

        Inputs a context-aware vector of a word and produces an analysis consists of root+tags

        Args:
            word_embedding (`torch.tensor`): word representations (outputs of char GRU)
            context_vector (`torch.tensor`): Context-aware representations of a words
            y (tuple): target tensors (encoded lemmas or encoded morph tags)

        Returns:
            `torch.tensor`: scores in each time step
        """

        # Initilize gru hidden units with context vector (encoder output)
        context_vectors = self.relu(self.W(context_vectors))
        hidden = torch.cat([context_vectors.view(1, *context_vectors.size()),
                            word_embeddings.view(1, *context_vectors.size())], 0)

        embeddings = self.embedding(y)
        embeddings = torch.relu(embeddings)
        outputs, _ = self.gru(embeddings, hidden)
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)

        return self.softmax(outputs)

    def predict(self, word_embedding, context_vector, max_len=50):
        """Forward pass of DecoderRNN for prediction only

        The loop for gru is stopped as soon as the end of sentence tag is produced twice.
        The first end of sentence tag indicates the end of the root while the second one indicates the end of tags

        Args:
            word_embedding (`torch.tensor`): word representation (outputs of char GRU
            context_vector (`torch.tensor`): Context-aware representation of a word
            max_len (int): Maximum length of produced analysis (Defaault: 50)

        Returns:
            tuple: (scores:`torch.tensor`, predictions:list)

        """

        # Initilize gru hidden units with context vector (encoder output)
        context_vector = context_vector.view(1, *context_vector.size())
        context_vector = self.relu(self.W(context_vector).view(1, 1, self.hidden_size))
        word_embedding = word_embedding.view(1, 1, self.hidden_size)
        hidden = torch.cat([context_vector, word_embedding], 0)

        # Oupput shape (maximum length of a an analyzer, output vocab size)
        scores = torch.zeros(max_len, self.vocab_size)

        # First predicted token is sentence start tag: 2
        predicted_token = torch.LongTensor(1).fill_(2)

        # Generate char or tag sequentially
        predictions = []
        for di in range(max_len):
            embedded = self.embedding(predicted_token).view(1, 1, -1)
            embedded = torch.relu(embedded)
            output, hidden = self.gru(embedded, hidden)
            output = self.classifier(output[0])
            scores[di] = self.softmax(output)
            topv, topi = output.topk(1)
            predicted_token = topi.squeeze().detach()
            # Increase eos count if produced output is eos
            if predicted_token.item() == 1:
                break
            # Add predicted output to predictions if it is not a special character such as eos or padding
            if predicted_token.item() > 2:
                predictions.append(self.index2token[predicted_token.item()])

        return scores, predictions


def test_encoder_decoder():
    train_data_path = '../data/2019/task2/UD_Afrikaans-AfriBooms/af_afribooms-um-train.conllu'
    from data_loaders import ConllDataset
    from torch.utils.data import DataLoader
    from train import predict

    train_set = ConllDataset(train_data_path, max_sentences=1)
    train_loader = DataLoader(train_set)

    encoder = EncoderRNN(10, 50, 50, len(train_set.surface_char2id))
    decoder_lemma = DecoderRNN(10, 50, train_set.lemma_char2id)
    decoder_morph_tags = DecoderRNN(10, 50, train_set.morph_tag2id)

    # Define loss and optimizers
    criterion = nn.NLLLoss(ignore_index=0)

    # Create optimizers
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_lemma_optimizer = torch.optim.Adam(decoder_lemma.parameters(), lr=0.001)
    decoder_morph_tags_optimizer = torch.optim.Adam(decoder_morph_tags.parameters(), lr=0.001)

    # Let the training begin
    for _ in tqdm(range(1000)):
        # Training part
        encoder.train()
        decoder_lemma.train()
        decoder_morph_tags.train()
        for ix, (x, y1, y2) in enumerate(train_loader):

            # Clear gradients for each sentence
            encoder.zero_grad()
            decoder_lemma.zero_grad()
            decoder_morph_tags.zero_grad()

            # Run encoder
            word_embeddings, context_embeddings = encoder(x)

            # Run decoder for each word
            sentence_loss = 0.0
            for _y, decoder in zip([y1, y2], [decoder_lemma, decoder_morph_tags]):
                decoder_outputs = decoder(word_embeddings, context_embeddings, _y[0, :, :-1])

                for word_ix in range(word_embeddings.size(0)):
                    sentence_loss += criterion(decoder_outputs[word_ix], _y[0, word_ix, 1:])

                sentence_loss.backward(retain_graph=True)

                # Optimization
                encoder_optimizer.step()
                decoder_lemma_optimizer.step()
                decoder_morph_tags_optimizer.step()

    encoder.eval()
    decoder_lemma.eval()
    decoder_morph_tags.eval()
    # Make predictions and save to file
    for sentence in train_set.sentences:
        surface_words = [surface_word for surface_word in sentence.surface_words]
        conll_sentence = predict(surface_words, encoder, decoder_lemma, decoder_morph_tags, train_set)
        print(conll_sentence)

if __name__ == '__main__':
    test_encoder_decoder()