import torch
import torch.nn as nn
import random


class EncoderRNN(nn.Module):
    """A bidirectional GRU as the context encoder.

    The inputs are encoded words in a sentence.
    EncoderRNN firstly apply bidirectional GRU to characters of words to build word representations
    Then apply a second level bidirectional GRU to the word representations.
    By doing so it generates context-aware representations of each word in a sentence which then be used in decoding.

    """

    def __init__(self, embedding_size, hidden_size1, hidden_size2, vocab_len, dropout_ratio=0.2):
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

        # Initilize hidden units of grus with Xavier initialization
        self.char_gru_hidden, self.word_gru_hidden = self.init_hidden()

    def init_hidden(self):
        """Initializes the hidden units of each gru

        """
        return (
            torch.zeros(1, 1, self.hidden_size1),
            torch.zeros(2, 1, self.hidden_size2)
        )

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

        self.char_gru_hidden, self.word_gru_hidden = self.init_hidden()

        # Embedding layer
        embeddeds = self.embedding(x)
        embeddeds = self.dropout(embeddeds)

        # First-level gru layer (char-gru to generate word embeddings)
        word_embeddings_list = []
        for i, char_embeddings in enumerate(embeddeds[0]):
            char_embeddings = char_embeddings.view(1, *char_embeddings.shape)
            outputs, _ = self.char_gru(char_embeddings, self.char_gru_hidden)
            word_embeddings_list.append(outputs[0][-1].view(1, 1, -1))
        word_embeddings = torch.cat(word_embeddings_list, 1)

        # Second-level gru layer (context-gru)
        context_embeddings = self.word_gru(word_embeddings, self.word_gru_hidden)[0]
        return word_embeddings, context_embeddings


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
        self.gru = nn.GRU(embedding_size, hidden_size, 2)
        self.classifier = nn.Linear(hidden_size, len(vocab))
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, word_embedding, context_vector, y, target_length, teacher_forcing_ratio=1.0):
        """Forward pass of DecoderRNN

        Inputs a context-aware vector of a word and produces an analysis consists of root+tags

        Args:
            word_embedding (`torch.tensor`): word representation (outputs of char GRU
            context_vector (`torch.tensor`): Context-aware representation of a word
            y (tuple): target tensors (encoded lemmas and encoded morph tags
            target_length (int): The length of the correct analysis
            teacher_forcing_ratio: probability to use teacher forcing

        Returns:
            `torch.tensor`: scores in each time step
        """
        # Decide whether to use teacher forcing
        if random.random() < teacher_forcing_ratio:
            use_teacher_forcing = True
        else:
            use_teacher_forcing = False

        # Initilize gru hidden units with context vector (encoder output)
        context_vector = context_vector.view(1, *context_vector.size())
        context_vector = self.relu(self.W(context_vector).view(1, 1, self.hidden_size))
        word_embedding = word_embedding.view(1, 1, self.hidden_size)
        hidden = torch.cat([context_vector, word_embedding], 0)

        # Oupput shape (maximum length of a an analyzer, output vocab size)
        scores = torch.zeros(target_length, self.vocab_size)

        # First predicted token is sentence start tag: 2
        predicted_token = torch.LongTensor(1).fill_(2)

        # Generate char or tag sequentially
        for di in range(target_length):
            # Propagate layers
            embedded = self.embedding(predicted_token).view(1, 1, -1)
            embedded = torch.relu(embedded)
            embedded = self.dropout(embedded)

            output, hidden = self.gru(embedded, hidden)
            output = self.dropout(output)
            output = self.classifier(output[0])

            # Save scores in each time step
            scores[di] = self.softmax(output)

            if use_teacher_forcing:
                # Use actual tag as input (teacher forcing)
                predicted_token = y[di]
            else:
                # Save the predicted token index which embedding layer inputs
                topv, topi = output.topk(1)
                predicted_token = topi.squeeze().detach()

        # Return the scores for each time step
        return scores

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
        eos_count = 0
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
                eos_count += 1
            # Add predicted output to predictions if it is not a special character such as eos or padding
            if predicted_token.item() > 2:
                predictions.append(self.index2token[predicted_token.item()])
            # If eos count greater than 1, stop iteration
            if eos_count >= 2:
                break

        return scores, predictions