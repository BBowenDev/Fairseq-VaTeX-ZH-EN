# Fairseq-VaTeX-ZH-EN
An implementation of fairseq trained on the VaTeX dataset's English (EN) and Simplified Chinese (ZH) parallel caption datasets.





## Installation


### Requirements
Fairseq-VaTeX-ZH-EN requires several packages. All are installed by scripts, but it may be helpful to keep track of installations by this repository.

* Python 3.0 or greater
* [Fairseq](https://github.com/pytorch/fairseq)
* [nltk](https://www.nltk.org/index.html)
* [punkt](https://github.com/nltk/nltk/blob/develop/nltk/tokenize/punkt.py)
* [jieba](https://github.com/fxsjy/jieba)
* [subword-nmt](https://github.com/rsennrich/subword-nmt)

To install `Fairseq-VaTeX-ZH-EN`, clone this repository and `cd` into it. Run the `prepare.sh` script to install the above requirements list and format folders. Depending on shell permissions, the script may need to be elevated with `chmod 755`.

```
git clone https://github.com/BraedenLB/Fairseq-VaTeX-ZH-EN.git
cd Fairseq-VaTeX-ZH-EN
#chmod 755 prepare.sh
bash prepare.sh
```

## Processing

### VaTeX Dataet
This implementation of Fairseq trains on text-only caption data from the [VaTeX dataset](https://eric-xw.github.io/vatex-website/index.html). The VaTeX datasets (`Training Set v1.0`, `Validaiton Set v1.0`, `Public Test Set v1.1`) are downloaded from the [VaTeX GitHub](https://eric-xw.github.io/vatex-website/download.html), tokenized, and processed in the `preprocess.sh` script.
In the VaTeX dataset, the final 5 captions of each video are parallel translations; to increase translation accuracy, only these parallel captions are used, for a total of 129,926 caption sentences per language.

### Preprocessing
The raw text JSON files are downloaded with `wget` and tokenized with the `vatex_preprocess.py` script. English (EN) captions are tokenized using [nltk](https://www.nltk.org/index.html)'s `word_tokenize`. Chinese (ZH) captions are tokenized using a [jieba](https://github.com/fxsjy/jieba). The jieba model utilized a pretrained, built-in dictionary to tokenize characters.
Tokenized `.en` and `.zh` files are created for each set `train`, `val`, and `test` and saved in `/vatex/tok`. 

Byte Pair Encoding (BPE) is used to encode tokenized caption data into a format readable by Fariseq. Because the BPE algorithm is language-independent, the same processing is done on both English (EN) and Chinese (ZH) datasets. The following implementations of BPE were tested: 
* [subword-nmt](https://github.com/rsennrich/subword-nmt)
* [Lei Mao's BPE](https://leimao.github.io/blog/Byte-Pair-Encoding/)
* [python-bpe](https://github.com/soaxelbrooke/python-bpe)
* [BPE Subword Tokennization](https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10)
* [Chinese-English NMT](https://github.com/twairball/fairseq-zh-en)
* [seq2seq](https://google.github.io/seq2seq/nmt/)

The implemented method uses a hybrid of [seq2seq](https://google.github.io/seq2seq/nmt/) and [Chinese-English NMT](https://github.com/twairball/fairseq-zh-en), which both implement [subword-nmt](https://github.com/rsennrich/subword-nmt). Other tested methods were faster or more thorough, but Fairseq expects data encoded by the `subword-nmt` package. 

To preprocess the VaTeX data, run the `preprocess.sh` script. Depending on shell permissions, the script may need to be elevated with `chmod 755`.

```
#chmod 755 preprocess.sh
bash preprocess.sh
```

## Training

## Testing & Evaluation
