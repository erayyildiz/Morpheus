<img src="https://raw.githubusercontent.com/erayyildiz/Morpheus/master/assets/Morpheus1.jpg" width="48">

# Morpheus: A Neural Network for Jointly Learning Contextual Lemmatization and Morphological Tagging

**Contextual Lemmatization and Morphological Tagging in 108 different languages. 
A Participant System for SigMorphon2019 Task 2**



## Introduction
*Morpheus* is a joint contextual lemmatizer and morphological tagger 
which is based on a neural sequential architecture where inputs are the characters of the surface words 
in a sentence and the outputs are the minimum edit operations between surface words and their lemmata as well as the
morphological tags assigned to the words. 

*Morpheus* does not rely on any language specific settings so it is able to run on any language without any effort.
According the results in SigMorphon 2019 Task 2, *Morpheus* performs comparable to current state-of-the-art systems 
for both lemmatization and morphological tagging tasks in nearly 100 languages.
*Morpheus* has placed 3rd in lemmatization and reached the 9th place in morphological tagging among all participant teams.

The experiments show that predicting edit actions instead of characters in the lemmata
is notably better, not only for lemmatization, but for tagging, as well.
The improvements especially in low resource languages are significant.

**The achitecture of Morpheus**

![Architecture](../master/assets/architecture.png?raw=true)

## Reference

Please cite the following work if you use the tool. 
The paper will be available in SIGMORPHON 2019 Proceedings.

> Eray Yildiz and A. Cuneyd Tantug. 2019. Morpheus: A Neural Network for Jointly Learning Contextual Lemmatization and Morphological Tagging. In Proceedings of the 16th SIGMORPHON Workshop on Computational Research in Phonetics,
Phonology, and Morphology, Florence, Italy. Association for Computational Linguistics

The paper can be found in [SIGMORPHON 2019 Proceedings](https://sigmorphon.github.io/workshops/2019/shipout-book.pdf)

## Datasets
The data is owes its provenance to the [Universal Dependencies](http://universaldependencies.org/) project and have been converted to the [UniMorph schema](https://unimorph.github.io/).

Sentences are annotated in the ten-column CoNLL-U format. 

- The ID column gives each word a unique ID within the sentence.
- The FORM column gives the word as it appears in the sentence.
- The LEMMA column contains the form’s lemma.
- The FEATS column contains morphosyntactic features in the UniMorph schema.
- 6 remaining columns are nulled out and replaced with undercore (‘_’)

At prediction time, test data also null out the LEMMA and FEATS columns. 

Check the [*UniMorph* dataset collection](https://github.com/sigmorphon/2019/tree/master/task2) which includes 
datasets for more than 100 languages.

## Usage

### Requirements
You can use any computer with `Python 3` installed.
We strongly recommend you to use a machine with a GPU if you want to train models.
To install dependencies, just install the packages written in `requirements.txt` as follow:

```bash
pip install -r requirements.txt

```

### Training
To train joint a contextual lemmatizer and morphological tagger for a language, run the following script in your command line.

```bash
train.py -l Turkish-IMST -t ../data/2019/task2/UD_Turkish-IMST/tr_imst-um-train.conllu 
-d ../data/2019/task2/UD_Turkish-IMST/tr_imst-um-dev.conllu -m my_model

```

This code will train a model for Turkish language. 

The options of the script as follow:

```bash


Options:
  -h, --help            show this help message and exit
  --all, --all          use if you want to train models for all languages in
                        UniMorph dataset
  -l LANGUAGE_NAME, --language_name=LANGUAGE_NAME
                        The name of the language
  -t TRAIN_FILE, --train_file=TRAIN_FILE
                        CONLL file path for training
  -d DEV_FILE, --dev_file=DEV_FILE
                        CONLL file path for validation
  -m MODEL_NAME, --model_name=MODEL_NAME
                        Name for the model

```

After placing the *UniMorph* datasets into the `data` directory, 
you can simply run following command to train models for all languages in the directory.

```bash
train.py --all -m my_model

```

After training has been completed, the following files are created in the same directory as the data:

- {{language_name}}.{{model_name}}.dataset
- {{language_name}}.{{model_name}}.encoder.model
- {{language_name}}.{{model_name}}.decoder_lemma.model
- {{language_name}}.{{model_name}}.decoder_morph.model

All of the created files will be used for prediction.

### Prediction

To run a model on a dataset, you can use the `predict.py` script as follow:

```bash
predict.py -i input_conll_file -o output_conll_file -d dataset_obj_path
-e encoder_model_file -l lemma_decoder_model_file -m morph_decoder_model_file

```

```bash
Options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file=INPUT_FILE
                        Input CONLL file path
  -o OUTPUT_FILE, --output_file=OUTPUT_FILE
                        Output file path
  -d DATASET_OBJ_FILE, --dataset_obj_file=DATASET_OBJ_FILE
                        The path of the dataset object which is saved during
                        training process
  -e ENCODER_FILE, --encoder_file=ENCODER_FILE
                        The path of the encoder object which is saved during
                        training process
  -l LEMMA_DECODER_FILE, --lemma_decoder_file=LEMMA_DECODER_FILE
                        The path of the lemma decoder object which is saved
                        during training process
  -m MORPH_DECODER_FILE, --morph_decoder_file=MORPH_DECODER_FILE
                        The path of the morph decoder object which is saved
                        during training process
```

**Note that the files DATASET_OBJ_FILE, LEMMA_DECODER_FILE and MORPH_DECODER_FILE are created during training**
The input file must be in conll format which is tab separated and the second column contains surface words.
The other columns are not imported and will be ignored.
The output file is also in conll format where third column contains lemmata and sixth column contains morphological tags.


### Experimental Results

| **Method** | **Lemmatization Accuracy (%)** | **Morphological Tagging F1 Score (%)** |
|--------------------------------------------------|----------------------------|:----------------------------------:|
| Turku NLP (Kanerva et al., 2018) | 92.18 | 86.7 |
| UPPSALA Uni. (Moor, 2018) | 58.5 | 88.32 |
| SigMorphon 2019 Baseline (Malaviya et al., 2019) | 93.95 | 68.72 |
| Morpheus (Character Prediction) | 88.03 | 88.94 |
| Morpheus (Edit Operation Prediction) | 94.15 | 90.52 |



| **Language** | **Dataset  Size** | **Lemmatization** |  | **Morphological  Tagging** |  |
|--------------------|:-------------:|:------------:|:-----------:|:---------------:|:------------:|
|  |  | **Character  Prediction Model** | **Edit  Prediction Model** | **Character  Prediction Model** | **Edit  Prediction Model** |
| North-Sami-Giella | 29K | 87.53 | 91.90 | 88.89 | 92.83 |
| French-GSD | 360K | 97.06 | 98.47 | 97.58 | 97.99 |
| Japanese-Modern | 14K | 85.39 | 93.88 | 93.06 | 92.44 |
| Swedish-PUD | 18K | 82.79 | 93.05 | 89.23 | 92.09 |
| Arabic-PADT | 256K | 94.39 | 95.18 | 95.01 | 95.40 |
| Basque-BDT | 119K | 95.42 | 96.49 | 93.06 | 94.47 |
| Urdu-UDTB | 123K | 95.20 | 96.02 | 90.79 | 91.20 |
| Irish-IDT | 21K | 85.07 | 89.23 | 80.60 | 71.52 |
| Bambara-CRB | 14K | 88.24 | 88.85 | 93.47 | 93.56 |
| Dutch-Alpino | 200K | 94.97 | 97.81 | 95.63 | 96.45 |
| Czech-FicTree | 175K | 97.39 | 98.49 | 94.15 | 96.39 |
| Danish-DDT | 94K | 93.16 | 97.26 | 94.17 | 95.62 |
| Latin-ITTB | 332K | 98.65 | 98.75 | 96.84 | 97.34 |
| French-Sequoia | 64K | 95.54 | 98.17 | 95.96 | 96.82 |
| Italian-PoSTWITA | 115K | 92.71 | 96.61 | 94.43 | 95.62 |
| Polish-SZ | 93K | 93.59 | 96.86 | 90.23 | 93.26 |
| Czech-CLTT | 32K | 92.11 | 98.03 | 89.03 | 93.82 |
| Cantonese-HK | 7K | 90.05 | 94.17 | 85.41 | 86.14 |
| Galician-TreeGal | 23K | 89.68 | 95.19 | 89.78 | 90.72 |
| Slovenian-SSJ | 131K | 95.25 | 96.94 | 93.47 | 95.79 |
| French-ParTUT | 25K | 92.67 | 96.10 | 93.09 | 94.55 |
| Lithuanian-HSE | 5K | 70.60 | 83.05 | 43.37 | 70.70 |
| French-Spoken | 35K | 94.47 | 98.48 | 95.46 | 96.66 |
| Russian-Taiga | 22K | 83.59 | 90.57 | 76.62 | 83.80 |
| Latvian-LVTB | 150K | 94.29 | 96.22 | 93.51 | 95.87 |
| Czech-PDT | 1515K | 84.86 | 98.41 | 87.65 | 95.27 |
| Japanese-GSD | 168K | 95.21 | 98.91 | 95.35 | 95.61 |
| Indonesian-GSD | 111K | 97.06 | 99.49 | 92.69 | 93.11 |
| Gothic-PROIEL | 62K | 96.60 | 96.58 | 93.04 | 95.12 |
| Latin-PROIEL | 219K | 96.31 | 96.37 | 93.75 | 95.05 |
| Czech-PUD | 19K | 83.55 | 93.57 | 81.30 | 86.70 |
| Dutch-LassySmall | 96K | 93.44 | 97.58 | 94.51 | 95.47 |
| Romanian-RRT | 198K | 96.54 | 97.88 | 96.81 | 97.44 |
| Korean-Kaist | 346K | 93.31 | 95.07 | 95.70 | 95.46 |
| Amharic-ATT | 11K | 93.80 | 100.00 | 91.02 | 91.39 |
| English-GUM | 79K | 95.58 | 97.85 | 93.92 | 95.48 |
| Estonian-EDT | 421K | 93.10 | 96.27 | 95.64 | 96.70 |
| Chinese-GSD | 111K | 95.22 | 99.15 | 89.25 | 90.78 |
| Korean-GSD | 80K | 87.55 | 92.89 | 93.43 | 94.16 |
| Marathi-UFAL | 4K | 74.59 | 76.94 | 68.26 | 69.26 |
| Akkadian | 2K | 42.22 | 60.89 | 78.13 | 66.52 |
| Faroese-OFT | 13K | 83.56 | 89.97 | 88.08 | 89.49 |
| English-EWT | 246K | 96.78 | 98.11 | 95.61 | 95.95 |
| Sanskrit-UFAL | 3K | 53.61 | 65.98 | 52.59 | 55.36 |
| Turkish-IMST | 60K | 94.13 | 96.43 | 91.67 | 93.72 |
| English-PUD | 20K | 89.40 | 95.22 | 88.88 | 89.89 |
| Korean-PUD | 18K | 87.19 | 98.86 | 91.42 | 92.75 |
| Finnish-PUD | 16K | 77.72 | 87.55 | 85.49 | 92.05 |
| Russian-SynTagRus | 1036K | 95.31 | 97.76 | 94.99 | 95.13 |
| Croatian-SET | 179K | 94.91 | 96.01 | 94.31 | 95.47 |
| Tagalog-TRG | 406 | 48.00 | 84.00 | 74.23 | 71.74 |
| Slovenian-SST | 31K | 91.83 | 94.97 | 85.34 | 89.23 |
| Finnish-FTB | 172K | 90.70 | 94.46 | 94.13 | 95.85 |
| Polish-LFG | 174K | 93.85 | 96.09 | 92.93 | 95.35 |
| Portuguese-Bosque | 218K | 96.43 | 97.86 | 96.07 | 96.59 |
| Coptic-Scriptorium | 20K | 93.47 | 95.68 | 95.17 | 94.76 |
| Chinese-CFL | 7K | 82.55 | 92.66 | 81.51 | 83.76 |
| Spanish-AnCora | 497K | 98.32 | 98.92 | 98.29 | 98.46 |
| Greek-GDT | 57K | 93.73 | 96.65 | 94.71 | 96.12 |
| Serbian-SET | 78K | 94.82 | 97.06 | 94.36 | 96.06 |
| Naija-NSC | 14K | 95.80 | 99.84 | 91.15 | 92.02 |
| Vietnamese-VTB | 42K | 98.17 | 99.95 | 89.45 | 89.71 |
| Yoruba-YTB | 2K | 83.60 | 97.20 | 80.49 | 70.67 |
| Italian-PUD | 22K | 89.51 | 96.11 | 92.63 | 94.22 |
| Finnish-TDT | 198K | 91.37 | 95.38 | 95.67 | 96.76 |
| English-ParTUT | 44K | 94.87 | 97.85 | 92.32 | 93.46 |
| Upper-Sorbian-U. | 11K | 77.79 | 90.74 | 69.47 | 77.46 |
| Norwegian-Ny. | 14K | 93.89 | 97.42 | 90.45 | 92.20 |
| Galician-CTG | 121K | 97.18 | 98.69 | 97.29 | 97.30 |
| Old-Church-Slv. | 66K | 96.48 | 95.66 | 93.33 | 94.91 |
| Russian-GSD | 92K | 92.90 | 91.51 | 91.98 | 93.91 |
| Kurmanji-MG | 10K | 85.66 | 92.69 | 85.99 | 85.22 |
| Norwegian-Bk. | 299K | 96.65 | 98.94 | 96.75 | 97.41 |
| Italian-ISDT | 273K | 96.90 | 97.90 | 97.34 | 97.89 |
| Komi-Zyrian-IKDP | 1K | 38.55 | 68.67 | 45.50 | 36.89 |
| Hebrew-HTB | 144K | 96.49 | 97.35 | 95.35 | 95.70 |
| Tamil-TTB | 10K | 86.77 | 96.10 | 83.07 | 88.50 |
| Buryat-BDT | 10K | 83.33 | 89.61 | 78.24 | 82.30 |
| Breton-KEB | 12K | 85.61 | 92.81 | 88.65 | 90.12 |
| Latin-Perseus | 29K | 87.30 | 86.26 | 79.21 | 82.88 |
| Romanian-Nonstd | 189K | 96.10 | 96.37 | 95.52 | 96.36 |
| Italian-ParTUT | 50K | 94.65 | 97.44 | 94.83 | 96.30 |
| Catalan-AnCora | 481K | 98.17 | 98.92 | 98.42 | 98.65 |
| Arabic-PUD | 22K | 81.31 | 80.90 | 87.23 | 88.00 |
| Komi-Zyrian-L. | 2K | 52.75 | 77.47 | 55.79 | 57.02 |
| Japanese-PUD | 25K | 86.30 | 97.32 | 93.46 | 94.02 |
| Slovak-SNK | 119K | 94.52 | 96.95 | 91.88 | 94.55 |
| Ukrainian-IU | 118K | 93.63 | 96.80 | 91.24 | 93.68 |
| Turkish-PUD | 17K | 78.20 | 89.19 | 86.71 | 91.28 |
| Bulgarian-BTB | 152K | 95.98 | 97.58 | 97.16 | 97.83 |
| Russian-PUD | 19K | 83.78 | 92.54 | 81.73 | 87.79 |
| Belarusian-HSE | 8K | 78.06 | 89.87 | 69.39 | 71.95 |
| Hindi-HDTB | 322K | 98.15 | 98.82 | 96.12 | 96.60 |
| Czech-CAC | 474K | 98.01 | 98.86 | 96.52 | 97.54 |
| Hungarian-Szeged | 38K | 87.89 | 95.26 | 89.63 | 91.65 |
| Swedish-LinES | 74K | 93.52 | 96.82 | 93.25 | 94.88 |
| Afrikaans-Af.B. | 45K | 93.75 | 98.74 | 95.08 | 95.96 |
| English-LinES | 77K | 96.19 | 98.27 | 94.57 | 95.43 |



## Acknowledgement
This work is carried out by Eray Yildiz and A. Cuneyd Tantug in Istanbul Technical University.
For questions: yildiz17@itu.edu.tr

We would like to thank the SigMorphon 2019 organizers for the great effort and the reviewers for
the insightful comments.
