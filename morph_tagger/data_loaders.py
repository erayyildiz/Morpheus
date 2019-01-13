import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data_utils import read_dataset


class ConllDataset(Dataset):
    """Torch dataset for sigmorphon conll data"""

    def __init__(self, conll_file_path, surface_char2id=None, lemma_char2id=None, morph_tag2id=None, mode='train'):
        """Initialize ConllDataset.

        :param conll_file_path: (str) conll file path
        :param surface_char2id: (dict) Default is None. if None calculated over given data
        :param lemma_char2id: (dict) Default is None. if None calculated over given data
        :param morph_tag2id: (dict) Default is None. if None calculated over given data
        :param mode: (str) 'train' or 'test'. If 'test' vocab dicts will not be updated
        """
        self.sentences = read_dataset(conll_file_path)
        if surface_char2id:
            self.surface_char2id = surface_char2id
        else:
            self.surface_char2id = dict()
        if lemma_char2id:
            self.lemma_char2id = lemma_char2id
        else:
            self.lemma_char2id = dict()
        if morph_tag2id:
            self.morph_tag2id = morph_tag2id
        else:
            self.morph_tag2id = dict()
        self.mode = mode

    def encode(self, seq, vocab):
        res = []
        for token in seq:
            if token in vocab:
                res.append(vocab[token])
            elif self.mode == 'train':
                vocab[token] = len(vocab)
                res.append(vocab[token])
        return torch.tensor(res, dtype=torch.long)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        max_token_len = max([len(surface) for surface in sentence.surface_words])
        max_lemma_len = max([len(lemma) for lemma in sentence.lemmas])
        max_morph_tags_len = max([len(morph_tag) for morph_tag in sentence.morph_tags])

        # Encode surfaces
        encoded_surfaces = torch.zeros((len(sentence), max_token_len), dtype=torch.long)
        for ix, surface in enumerate(sentence.surface_words):
            encoded_surfaces[ix, :len(surface)] = self.encode(surface, self.surface_char2id)

        # Encode lemmas
        encoded_lemmas = torch.zeros((len(sentence), max_lemma_len), dtype=torch.long)
        for ix, lemma in enumerate(sentence.lemmas):
            encoded_lemmas[ix, :len(lemma)] = self.encode(lemma, self.lemma_char2id)

        # Encode surfaces
        encoded_morph_tags = torch.zeros((len(sentence), max_morph_tags_len), dtype=torch.long)
        for ix, morph_tag in enumerate(sentence.morph_tags):
            encoded_morph_tags[ix, :len(morph_tag)] = self.encode(morph_tag, self.morph_tag2id)

        return encoded_surfaces, encoded_lemmas, encoded_morph_tags


dataset = ConllDataset('../data/2019/task2/UD_Turkish-IMST/tr_imst-um-dev.conllu')
data_loader = DataLoader(dataset)

for x, y1, y2 in data_loader:
    print(x.size())
    print(y1.size())
    print(y2.size())
    break