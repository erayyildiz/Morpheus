from itertools import chain

import torch
from torch.utils.data import Dataset
from data_utils import read_dataset


class ConllDataset(Dataset):
    """Torch dataset for sigmorphon conll data"""

    PAD_token = '<p>'
    EOS_token = '<e>'

    def __init__(self, conll_file_path, surface_char2id=None, lemma_char2id=None, morph_tag2id=None,
                 mode='train', max_sentences=0):
        """Initialize ConllDataset.

        Arguments:
            conll_file_path (str): conll file path
            surface_char2id (dict): Default is None. if None calculated over given data
            lemma_char2id (dict): Default is None. if None calculated over given data
            morph_tag2id (dict): Default is None. if None calculated over given data
            mode (str): 'train' or 'test'. If 'test' vocab dicts will not be updated
            max_sentences (int): Maximum number of sentences to be loaded into dataset.
                Default is 0 which means no limitation
        """
        self.sentences = read_dataset(conll_file_path)
        if 0 < max_sentences < len(self.sentences):
            self.sentences = self.sentences[:max_sentences]
        if surface_char2id:
            self.surface_char2id = surface_char2id
        else:
            self.surface_char2id = dict()
            self.surface_char2id[self.PAD_token] = len(self.surface_char2id)
            self.surface_char2id[self.EOS_token] = len(self.surface_char2id)
        if lemma_char2id:
            self.lemma_char2id = lemma_char2id
        else:
            self.lemma_char2id = dict()
            self.lemma_char2id[self.PAD_token] = len(self.lemma_char2id)
            self.lemma_char2id[self.EOS_token] = len(self.lemma_char2id)
        if morph_tag2id:
            self.morph_tag2id = morph_tag2id
        else:
            self.morph_tag2id = dict()
            self.morph_tag2id[self.PAD_token] = len(self.morph_tag2id)
            self.morph_tag2id[self.EOS_token] = len(self.morph_tag2id)
        self.mode = mode
        if mode == 'train':
            self.create_vocabs()

    def create_vocabs(self):
        """Create surface_char2id, lemma_char2id and morph_tag2id vocabs using provided data

        """
        # Update surface_char2id
        unique_surfaces = set(chain(*[sentence.surface_words for sentence in self.sentences]))
        unique_chars = set(chain(*[surface for surface in unique_surfaces]))
        for ch in unique_chars:
            self.surface_char2id[ch] = len(self.surface_char2id)

        # Update lemma_char2id
        unique_lemmas = set(chain(*[sentence.lemmas for sentence in self.sentences]))
        unique_chars = set(chain(*[lemma for lemma in unique_lemmas]))
        for ch in unique_chars:
            self.lemma_char2id[ch] = len(self.lemma_char2id)

        # Update morph_tag2id
        unique_morph_tags = list(chain(*[sentence.morph_tags for sentence in self.sentences]))
        unique_tags = set(chain(*[morph_tag for morph_tag in unique_morph_tags]))
        for tag in unique_tags:
                self.morph_tag2id[tag] = len(self.morph_tag2id)

    @staticmethod
    def encode(seq, vocab):
        res = []
        for token in seq:
            if token in vocab:
                res.append(vocab[token])
        res.append(vocab[ConllDataset.EOS_token])
        return torch.tensor(res, dtype=torch.long)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        max_token_len = max([len(surface)+1 for surface in sentence.surface_words])
        max_lemma_len = max([len(lemma)+1 for lemma in sentence.lemmas])
        max_morph_tags_len = max([len(morph_tag)+1 for morph_tag in sentence.morph_tags])

        # Encode surfaces
        encoded_surfaces = torch.zeros((len(sentence), max_token_len), dtype=torch.long)
        for ix, surface in enumerate(sentence.surface_words):
            encoded_surface = self.encode(surface, self.surface_char2id)
            encoded_surfaces[ix, :encoded_surface.size()[0]] = encoded_surface

        # Encode lemmas
        encoded_lemmas = torch.zeros((len(sentence), max_lemma_len), dtype=torch.long)
        for ix, lemma in enumerate(sentence.lemmas):
            encoded_lemma = self.encode(lemma, self.lemma_char2id)
            encoded_lemmas[ix, :encoded_lemma.size()[0]] = encoded_lemma

        # Encode surfaces
        encoded_morph_tags = torch.zeros((len(sentence), max_morph_tags_len), dtype=torch.long)
        for ix, morph_tag in enumerate(sentence.morph_tags):
            encoded_morph_tag = self.encode(morph_tag, self.morph_tag2id)
            encoded_morph_tags[ix, :encoded_morph_tag.size()[0]] = encoded_morph_tag

        return encoded_surfaces, encoded_lemmas, encoded_morph_tags