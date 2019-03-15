from itertools import chain
from Levenshtein.StringMatcher import editops


def find_transformation(surface, lemma):
    edits = editops(surface, lemma)
    # print(edits)
    l = len(surface)
    labels = ['same'] * l
    for edit_type, source_ix, target_ix in edits:
        if labels[source_ix] == 'same':
            if edit_type.startswith('insert'):
                labels[source_ix] = 'insert_' + lemma[target_ix]
            elif edit_type.startswith('replace'):
                labels[source_ix] = 'replace_' + lemma[target_ix]
            elif edit_type.startswith('delete'):
                labels[source_ix] = 'delete'
        elif labels[source_ix].startswith('insert') and edit_type.startswith('replace'):
            labels[source_ix] = labels[source_ix].replace('insert', 'replace') + lemma[target_ix]
        elif labels[source_ix].startswith('insert') or labels[source_ix].startswith('replace'):
            labels[source_ix] = labels[source_ix] + lemma[target_ix]
        else:
            raise IOError('Invalid transformation! surfac:{}, lemma:{}, transformation:{}'.format(
                surface, lemma, transformation)
            )

    assert len(labels) == len(surface), 'Surface and transformation length are different'
    return labels


def inverse_transformation(surface, edits):
    res = []
    surface_ix = 0
    for edit in edits:
        ch = surface[surface_ix]
        if edit == 'same':
            res.append(ch)
            surface_ix += 1
        elif edit == 'delete':
            surface_ix += 1
            continue
        elif edit.startswith('replace'):
            replacement = edit.replace('replace_', '')
            res.append(replacement)
            surface_ix += 1
        elif edit.startswith('insert'):
            replacement = edit.replace('insert_', '')
            res.append(replacement)
            res.append(ch)

            surface_ix += 1

    return ''.join(res)


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
        self.transformations = []

        for conll_token in conll_sentence:
            if not conll_token or conll_token.startswith('#'):
                continue
            _splits = conll_token.split('\t')
            self.surface_words.append(_splits[1] + '$')
            self.lemmas.append(_splits[2])
            self.transformations.append(find_transformation(_splits[1] + '$', _splits[2]))
            assert inverse_transformation(_splits[1] + '$', self.transformations[-1]) == _splits[2], \
                'Transformation incorrect: {} - {} - {}'.format(_splits[1], _splits[2], self.transformations[-1])
            self.morph_tags.append(_splits[5].split(';'))

    def get_tags_as_str(self):
        return [';'.join(morph_tags) for morph_tags in self.morph_tags]

    def __len__(self):
        return len(self.lemmas)

    def __repr__(self):
        return "\n".join(
            ['Surface: {}, Lemma: {}, MorphTags: {}'.format(surface, lemma, ';'.join(morph_tags))
             for surface, lemma, morph_tags in zip(self.surface_words, self.lemmas, self.morph_tags)]
        )


def read_dataset(conll_file):
    """Read Conll dataset

    Arguments:
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


def read_surfaces(conll_file):
    """Read surface words from a Conll dataset

    Arguments:
        conll_file: (str) conll file path
    Returns:
        list: list of sentences that consist of surface words
    """

    sentences = []
    with open(conll_file, 'r', encoding='UTF-8') as f:
        conll_sentence = []
        for line in f:
            if len(line.strip()) == 0:
                if len(conll_sentence) > 0:
                    sentence = []
                    for conll_token in conll_sentence:
                        if not conll_token or conll_token.startswith('#'):
                            continue
                        sentence.append(conll_token.split('\t')[1])
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