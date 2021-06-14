# Fairseq-VaTeX-ZH-EN
An implementation of fairseq for sequence-to-sequence translation trained on the VaTeX dataset's English (EN) and Simplified Chinese (ZH) parallel caption datasets in Google COLAB.

Although all captions in the VaTeX dataset reference the same images and are thus parallel, the last 5 captions (of 10 captions) per image are paired EN-ZH translations. To ensure greatest accuracy of translation, only these parallel captions are used. See **preprocessing** for more information on accessing the parallel captions.

**_NOTE:_** This documentation is designed to be replicated in Google COLAB, but the same processes may be followed with altered syntax for any Unix shell.

## Installation
This repository contains a preinstalled version of fairseq and all of the files required to train a text-only model with VaTeX captions. 

### Installing prerequisites 
The following libraries are required, but installed by default in the `preprocess.sh` script:
* fairseq
* nltk
* jieba
* 


### Installing VaTeX from the source repository
**OPTIONAL:** This repository contains the files necessary to preprocess the VaTeX captions, but the originals (including the Training set, Validation set, and Test sets) can be accessed [here](https://eric-xw.github.io/vatex-website/download.html).


### Installing this repository
This repository contains all of the necessary VaTeX and Fairseq files in the `raw` directory (`vatex_public_test_english_v1.1.json`, `vatex_training_v1.0.json`, `vatex_validation_v1.0.json`). For the simplest installation, `git clone` this repository and run the `prepare.sh` script. 

Depending on existing permissions, the script may need to be modified with `chmod 755`.
```
%cd /root/
!git clone https://github.com/BraedenLB/Fairseq-VaTeX-ZH-EN.git
%cd /root/Fairseq-VaTeX-ZH-EN
#!chmod 755 ./prepare.sh
!./prepare.sh
```

## Preprocessing
