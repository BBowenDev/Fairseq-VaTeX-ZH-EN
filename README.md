# Fairseq-VaTeX-ZH-EN
An implementation of fairseq for sequence-to-sequence translation trained on the VaTeX dataset's English (EN) and Simplified Chinese (ZH) parallel caption datasets in Google COLAB.

In the VaTeX dataset, the final 5 captions for each video are directly parallel between languages. To ensure greatest accuracy of translation, only these parallel captions are used. See **preprocessing** for more information on accessing the parallel captions.

**_NOTE:_** This documentation is designed to be replicated in Google COLAB, but the same processes may be followed with altered syntax for any Unix shell.

## Installation
This repository contains a preinstalled version of fairseq and all of the files required to train a text-only model with VaTeX captions. 

### Installing VaTeX from the source repository
**OPTIONAL:** This repository contains the files necessary to preprocess the VaTeX captions, but the originals (including the Training set, Validation set, and Test sets) can be accessed [here](https://eric-xw.github.io/vatex-website/download.html).


### Installing this repository
This repository contains all of the necessary VaTeX and Fairseq files. For the simplest installation, `git clone` this repository.

```
%cd /root/
!git clone <Fairseq-VaTeX-ZH-EN>
%cd /root/Fairseq-VaTeX-ZH-EN
!mkdir tok
```
  
## Preprocessing
