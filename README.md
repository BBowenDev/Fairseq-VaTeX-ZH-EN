# Fairseq-VaTeX-ZH-EN
A text-only implementation of fairseq trained on the VaTeX dataset's English (EN) and Simplified Chinese (ZH) caption datasets.

## Installation
This model relies on pre-built shell scripts for installation. These scripts will run without necessary modification, but modifications can be made to optimize or customize installation.

### Requirements
Fairseq-VaTeX-ZH-EN requires several packages. All are installed by scripts, but it may be helpful to keep track of installations by this repository.

* Python 3.0 or greater
* [Fairseq](https://github.com/pytorch/fairseq)
* [nltk](https://www.nltk.org/index.html)
* [punkt](https://github.com/nltk/nltk/blob/develop/nltk/tokenize/punkt.py)
* [sacremoses](https://github.com/alvations/sacremoses)
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

In the VaTeX dataset, the final 5 captions of each video are parallel translations. These parallel captions can be selected for use, but the full dataset is used by default. Additionally, the `Public Test Set v1.1` set does not contain any Simplified Chinese (ZH), so 10,000 English (EN) and Chinese (ZH) captions are taken from the `Validaiton Set v1.0` set to test the model.

The `vatex_preprocess.py` script has optional arguments to modify dataset usage:
* `-f True` [True/False] if True, all captions. If False, utilize only parallel translations 
* `-t 10000` [int] number of captions from validation set to be used for testing (maximum 29,999)

**_NOTE_**: The small size of the validation and testing sets may raise [warnings](https://github.com/rsennrich/subword-nmt/issues/63) from Fairseq, but the model will still train and test.

### Tokenization
The raw text JSON files are downloaded with `wget` and tokenized with the `vatex_preprocess.py` script in the `preprocess.sh` script. English (EN) captions are tokenized using [nltk](https://www.nltk.org/index.html)'s `word_tokenize`. Chinese (ZH) captions are tokenized using [jieba](https://github.com/fxsjy/jieba)'s `lcut`. The jieba model utilized a pretrained, built-in dictionary to tokenize characters.

Tokenized `.en` and `.zh` files are created for each set `train`, `val`, and `test` and saved in `/vatex/tok`. 
* [mosesdecoder](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) was also tested as a replacement for nltk, bit it was found that it unecessarily increased the complexity of tokenization.

### Byte Pair Encoding
Byte Pair Encoding (BPE) is used to encode tokenized caption data into a format readable by Fariseq. Because the BPE algorithm is language-independent, the same processing is done on both English (EN) and Chinese (ZH) datasets. The following implementations of BPE were tested: 
* [subword-nmt](https://github.com/rsennrich/subword-nmt)
* [Lei Mao's BPE](https://leimao.github.io/blog/Byte-Pair-Encoding/)
* [python-bpe](https://github.com/soaxelbrooke/python-bpe)
* [BPE Subword Tokennization](https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10)
* [Chinese-English NMT](https://github.com/twairball/fairseq-zh-en)
* [seq2seq](https://google.github.io/seq2seq/nmt/)

The implemented method uses a hybrid of [seq2seq](https://google.github.io/seq2seq/nmt/) and [Chinese-English NMT](https://github.com/twairball/fairseq-zh-en), which both implement [subword-nmt](https://github.com/rsennrich/subword-nmt). Other tested methods were faster or more thorough, but Fairseq expects data encoded by the `subword-nmt` package. 

### Preprocessing
To preprocess the VaTeX data, run the `preprocess.sh` script. Depending on shell permissions, the script may need to be elevated with `chmod 755`.

```
#chmod 755 preprocess.sh
bash preprocess.sh
```

Once preprocessing has completed, define the locations of BPE files.

| **Variable** | **Location** |
|------------|------------|
| TRAIN      | ../Fairseq-VaTeX-ZH-EN/vatex/bpe/train.bpe10000 |
| VAL        | ../Fairseq-VaTeX-ZH-EN/vatex/bpe/val.bpe10000 |
| TEST       | ../Fairseq-VaTeX-ZH-EN/vatex/bpe/test.bpe10000 |
| DATADIR    | ../Fairseq-VaTeX-ZH-EN/vatex/data |
| MODELS     | ../Fairseq-VaTeX-ZH-EN/models |

Then, run `fairseq-preprocess` with the following arguments:

```
cd ../Fairseq-VaTeX-ZH-EN
fairseq-preprocess --source-lang zh --target-lang en --trainpref $TRAIN --validpref $VAL --destdir $DATADIR --testpref $TEST --workers 20
```

## Training

This model trains using the [`lightconv_wmt_zh_en_big`](https://github.com/pytorch/fairseq/blob/master/examples/pay_less_attention_paper/README.md) ZH-EN model architecture.

To train the model, run `fairseq-train` with the arguments below. Depending on system settings, directory variables (e.g., `$MODELS`) may need to be replaced with directory strings (e.g., `"../Fairseq-VaTeX-ZH-EN"`).

```
cd ../Fairseq-VaTeX-ZH-EN
fairseq-train $DATADIR --task translation --arch lightconv_wmt_zh_en_big --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 600 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --max-epoch 10 --save-dir $MODELS --save-interval 1
```

## Testing & Evaluation

Once the model has been trained, it can be tested with `fairseq-generate` with the arguments below. Depending on system settings, directory variables (e.g., `$DATADIR`) may need to be replaced with directory strings (e.g., `"../Fairseq-VaTeX-ZH-EN/vatex/data"`).

```
cd ../Fairseq-VaTeX-ZH-EN
fairseq-generate $DATADIR --task translation --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --path $MODELS/checkpoint_best.pt
```
