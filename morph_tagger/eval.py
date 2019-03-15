"""Evaluation for the SIGMORPHON 2019 shared task, task 2.
Computes various metrics on input.

"""

import argparse
import logging
import os
import numpy as np

from collections import namedtuple
from pathlib import Path

from languages import LANGUAGES
from logger import LOGGER
import pandas as pd

log = logging.getLogger(Path(__file__).stem)

COLUMNS = "ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC".split()
ConlluRow = namedtuple("ConlluRow", COLUMNS)
SEPARATOR = ";"


def distance(str1, str2):
    """Simple Levenshtein implementation."""
    m = np.zeros([len(str2)+1, len(str1)+1])
    for x in range(1, len(str2) + 1):
        m[x][0] = m[x-1][0] + 1
    for y in range(1, len(str1) + 1):
        m[0][y] = m[0][y-1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y-1] == str2[x-1]:
                dg = 0
            else:
                dg = 1
            m[x][y] = min(m[x-1][y] + 1, m[x][y-1] + 1, m[x-1][y-1] + dg)
    return int(m[len(str2)][len(str1)])


def set_equal(str1, str2):
    set1 = set(str1.split(SEPARATOR))
    set2 = set(str2.split(SEPARATOR))
    return set1 == set2


def manipulate_data(pairs):
    log.info("Lemma acc, Lemma Levenshtein, morph acc, morph F1")

    count = 0
    lemma_acc = 0
    lemma_lev = 0
    morph_acc = 0

    f1_precision_scores = 0
    f1_precision_counts = 0
    f1_recall_scores = 0
    f1_recall_counts = 0

    for r, o in pairs:
        log.debug("{}\t{}\t{}\t{}".format(r.LEMMA, o.LEMMA, r.FEATS, o.FEATS))
        count += 1
        lemma_acc += (r.LEMMA == o.LEMMA)
        lemma_lev += distance(r.LEMMA, o.LEMMA)
        morph_acc += set_equal(r.FEATS, o.FEATS)

        r_feats = set(r.FEATS.split(SEPARATOR)) - {"_"}
        o_feats = set(o.FEATS.split(SEPARATOR)) - {"_"}

        union_size = len(r_feats & o_feats)
        reference_size = len(r_feats)
        output_size = len(o_feats)

        f1_precision_scores += union_size
        f1_recall_scores += union_size
        f1_precision_counts += output_size
        f1_recall_counts += reference_size

    f1_precision = f1_precision_scores / (f1_precision_counts or 1)
    f1_recall = f1_recall_scores / (f1_recall_counts or 1)
    f1 = 2 * (f1_precision * f1_recall) / (f1_precision + f1_recall + 1E-20)

    return (100 * lemma_acc / count, lemma_lev / count, 100 * morph_acc / count, 100 * f1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--reference',
                        type=Path)
    parser.add_argument('-o', '--output',
                        type=Path)
    # Set the verbosity level for the logger. The `-v` option will set it to
    # the debug level, while the `-q` will set it to the warning level.
    # Otherwise use the info level.
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('-v', '--verbose', action='store_const',
                           const=logging.DEBUG, default=logging.INFO)
    verbosity.add_argument('-q', '--quiet', dest='verbose',
                           action='store_const', const=logging.WARNING)
    return parser.parse_args()


def strip_comments(lines):
    for line in lines:
        if not line.startswith("#"):
            yield line


def read_conllu(file: Path):
    with open(file) as f:
        yield from strip_comments(f)


def input_pairs(reference, output):
    for r, o in zip(reference, output):
        assert r.count("\t") == o.count("\t"), (r.count("\t"), o.count("\t"), o)
        if r.count("\t") > 0:
            r_conllu = ConlluRow._make(r.split("\t"))
            o_conllu = ConlluRow._make(o.split("\t"))
            yield r_conllu, o_conllu


def evaluate(language_name, language_path, model_name=None):
    LOGGER.info('Reading files for language: {}'.format(language_name))
    language_conll_files = os.listdir(language_path)

    for language_conll_file in language_conll_files:
        if 'dev.' in language_conll_file:
            val_data_path = language_path + '/' + language_conll_file
            if model_name:
                prediction_file = val_data_path.replace('dev', 'predictions-{}'.format(model_name))
            else:
                prediction_file = val_data_path.replace('dev', 'predictions')
            reference = read_conllu(val_data_path)
            output = read_conllu(prediction_file)
            cur_results = manipulate_data(input_pairs(reference, output))

            LOGGER.info('Evaluation completed')
            LOGGER.info('Lemma Acc:{}, Lemma Lev. Dist: {}, Morph acc: {}, F1: {} '.format(*cur_results))

            return {
                'Language': language_name,
                'Language Code': LANGUAGES[language_name][0],
                'Baseline Lemma Acc': float(LANGUAGES[language_name][1]),
                'Baseline Lemma Lev. Dist': float(LANGUAGES[language_name][2]),
                'Baseline Morph Acc': float(LANGUAGES[language_name][3]),
                'Baseline Morph F1': float(LANGUAGES[language_name][4]),
                'Lemma Acc': cur_results[0],
                'Lemma Lev. Dist': cur_results[1],
                'Morph Acc': cur_results[2],
                'Morph F1': cur_results[3],
            }


def evaluate_all(model_name=None):
    data_path = '../data/2019/task2/'
    language_paths = [data_path + filename for filename in os.listdir(data_path)]
    language_names = [filename.replace('UD_', '') for filename in os.listdir(data_path)]

    results = []

    for language_path, language_name in zip(language_paths, language_names):
        try:
            results.append(evaluate(language_name, language_path, model_name=model_name))
        except Exception as e:
            LOGGER.error(e)

    LOGGER.info('Writing results to file...')
    df = pd.DataFrame(results, columns=['Language Code', 'Language', 'Baseline Lemma Acc',
                                        'Baseline Lemma Lev. Dist', 'Baseline Morph Acc', 'Baseline Morph F1',
                                        'Lemma Acc', 'Lemma Lev. Dist', 'Morph Acc', 'Morph F1'],
                      index=None)
    df.to_excel('Results.xlsx')

if __name__ == "__main__":
    # evaluate_all()
    evaluate('Turkish-PUD', '../data/2019/task2/UD_Turkish-PUD', model_name='standard_morphnet')