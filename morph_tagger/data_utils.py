from itertools import chain


class Sentence(object):
    """Sentence class with surface words, lemmas and morphological tags

    """

    def __init__(self, conll_sentence):
        """Create a Sentence object from a conll sentence

        Arguments:
            conll_sentence: (list) list of conll lines correspond to one sentence
        """
        self.surface_words = []
        self.lemmas = []
        self.morph_tags = []

        for conll_token in conll_sentence:
            if not conll_token or conll_token.startswith('#'):
                continue
            _splits = conll_token.split('\t')
            self.surface_words.append(_splits[1])
            self.lemmas.append(_splits[2])
            self.morph_tags.append(_splits[5].split(';'))

    def get_tags_as_str(self):
        return [';'.join(morph_tags) for morph_tags in self.morph_tags]

    def __repr__(self):
        return "\n".join(
            ['Surface: {}, Lemma: {}, MorphTags: {}'.format(surface, lemma, ';'.join(morph_tags))
             for surface, lemma, morph_tags in zip(self.surface_words, self.lemmas, self.morph_tags)]
        )


def read_dataset(conll_file):
    """Read Conll dataset

    Argumets:
        conll_file: (str) conll file path
    Returns:
        list: list of `Sentence` objects
    """
    sentences = []
    with open(conll_file, 'r', encoding='UTF-8') as f:
        conll_sentence = []
        for line in f:
            if len(line.strip()) == 0:
                if len(conll_sentence) > 0:
                    sentence = Sentence(conll_sentence)
                    sentences.append(sentence)
                conll_sentence = []
            else:
                conll_sentence.append(line)
    return sentences


def get_stats(sentences):
    """Calculate statistics of surface words, lemmas and morphological tags in given sentences

    Arguments:
        sentences: (list) list of `Sentence` objects
    Returns:
        dict: stats dict
    """

    def flatten(_list):
        return list(chain(*_list))

    number_of_sentences = len(sentences)
    number_of_tokens = len(flatten([sentence.surface_words for sentence in sentences]))
    number_of_unique_words = len(set(flatten([sentence.surface_words for sentence in sentences])))
    number_of_unique_lemmas = len(set(flatten([sentence.lemmas for sentence in sentences])))
    number_of_unique_tags = len(set(flatten([sentence.get_tags_as_str() for sentence in sentences])))
    number_of_unique_features = len(set(flatten(flatten([sentence.morph_tags for sentence in sentences]))))

    return {
        'Number of sentence': number_of_sentences,
        'Number of tokens': number_of_tokens,
        'Number of unique words': number_of_unique_words,
        'Number of unique lemmas': number_of_unique_lemmas,
        'Number of unique morphological tags': number_of_unique_tags,
        'Number of unique morphological features': number_of_unique_features
    }