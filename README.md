Multimodal machine translation using [fairseq](https://github.com/pytorch/fairseq).

## Installation

Make sure to have CUDA 10.2 to be compatible with Pytorch 1.7. This was tested with Python 3.6.

Git clone and update submodules (`git clone $URL` and then `git submodule update --init --recursive`.

Create a new environment (`python3 -m venv env/` or `pipenv --python 3.6` if you are using pipenv) and activate the environment (`source env/bin/activate` or `pipenv shell`).

Install the requirements (`pip install requirements.txt` or `pipenv install`).

Install dependencies (`bash install.sh`).

## Training multimodal Transformer model on the Multi30k dataset

See [here](https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md) for instructions on how to train a regular Transformer model on the IWSLT'14 dataset. 
Being familiar with those instructions makes it easy to follow these instrucitons.


Let `$ROOT` be the root directory of this repo.
Create the `$ROOT/work/` directory and `cd` to the directory.

### Data pre-processing

Here, we create the en-de Multi30k dataset for translating from English (en) to German (de).

#### Text pre-processing

Download the Mutlti30k dataset (from `https://github.com/vvjn/multi30k-wmt18`) into the `$ROOT/work/data/multi30k-wmt18/` directory.
Download the images into the `$ROOT/work/data/flickr30k-images`.
Follow instructions in that repo to tokenize and BPE encode the text, and to download the raw images.

```
TEXT=data/multi30k-wmt18/task1-data
DATADIR=data-bin/multi30k.en-de.lc.norm.tok.bpe10000
fairseq-preprocess --source-lang en --target-lang de --trainpref $TEXT/en-de/train.lc.norm.tok.bpe10000 --validpref $TEXT/en-de/val.lc.norm.tok.bpe10000 --testpref $TEXT/en-de/test_2016_flickr.lc.norm.tok.bpe10000 --destdir $DATADIR --workers 20
```

#### Image pre-processing

Extract global image features using ResNet50. The pre-trained model is taken from (`https://download.pytorch.org/models/resnet50-0676ba61.pth`).

```
IMAGEDIR=data/flickr30k-images
MODELFILE=models/resnet50-0676ba61.pth
FEATSDIR=data/multi30k-wmt18/task1-data/feats
TESTNAME=test_2016_flickr

python src/extract_image_feats_resnet50.py --image-folder $IMAGEDIR --file-names $TEXT/image_splits/train.txt --batch-size 256 --model-file $MODELFILE --output-prefix $FEATSDIR/train
python src/extract_image_feats_resnet50.py --image-folder $IMAGEDIR --file-names $TEXT/image_splits/val.txt --batch-size 256 --model-file $MODELFILE --output-prefix $FEATSDIR/valid
python src/extract_image_feats_resnet50.py --image-folder $IMAGEDIR --file-names $TEXT/image_splits/${TESTNAME}.txt --batch-size 256 --model-file $MODELFILE --output-prefix $FEATSDIR/test
```

### Training the model

#### Text-only training

```
FAIRSEQDIR=../src/mmt
MODELSTEXTONLY=checkpoints/transformer_iwslt_de_en-multi30k.en-de.textonly
fairseq-train $DATADIR --user-dir $FAIRSEQ --task translation --arch transformer_iwslt_de_en --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 600 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --max-epoch 40 --save-dir $MODELSTEXTONLY --save-interval 10
```

#### Multimodal training

Using ResNet50's average pooling layer output.

```
MODELSAVGPOOL=checkpoints/imgc_transformer_iswlt_de_en_decadd-multi30k.en-de.avgpool
fairseq-train $DATADIR --user-dir $FAIRSEQDIR --task multimodal_translation --arch imgc_transformer_iwslt_de_en --feat-path $FEATSDIR --feats-suffix '_resnet50-avgpool.npy' --feats-shape '(2048,)' --feats-combine 'dec+add' --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 600 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --max-epoch 40 --save-dir $MODELSAVGPOOL --save-interval 10
```

Using ResNet50's res4f layer output.

```
MODELSRES4FRELU=checkpoints/imgc_transformer_iswlt_de_en_decadd-multi30k.en-de.res4frelu
fairseq-train $DATADIR --user-dir $FAIRSEQDIR --task multimodal_translation --arch imgc_transformer_iwslt_de_en --feat-path $FEATSDIR --feats-suffix '_resnet50-res4frelu.npy' --feats-shape '(1024,14,14)' --feats-combine 'dec+add' --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 600 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --max-epoch 40 --save-dir $MODELSRES4FRELU --save-interval 10
```

### Testing and evaluation

#### Text-only
```
fairseq-generate $DATADIR --user-dir $FAIRSEQDIR --task translation --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --path $MODELSTEXTONLY/checkpoint_best.pt
```

#### Multimodal

Using ResNet50's average pooling layer output.

```
fairseq-generate $DATADIR --user-dir $FAIRSEQDIR --task multimodal_translation --feats-path $FEATSDIR --feats-suffix '_resnet50-avgpool.npy' --feats-test-name $TESTNAME --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --path $MODELSAVGPOOL/checkpoint_best.pt
```

Using ResNet50's res4f layer output.

```
fairseq-generate $DATADIR --user-dir $FAIRSEQDIR --task multimodal_translation --feats-path $FEATSDIR --feats-suffix '_resnet50-res4frelu.npy' --feats-test-name $TESTNAME --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --path $MODELSRES4FRELU/checkpoint_best.pt
```

### Testing and evaluation on a new dataset

#### Text preprocessing

```
TEXTNEW=data/multi30k-wmt18/task1-data
DATADIRNEW=data-bin/multi30k_test_2017_flickr.en-de.lc.norm.tok.bpe10000
TESTNAMENEW=test_2017_flickr
fairseq-preprocess --source-lang en --target-lang de --srcdict $DATADIR/dict.en.txt --tgtdict $DATADIR/dict.de.txt --testpref $TEXTNEW/en-de/${TESTNAMENEW}.lc.norm.tok.bpe10000 --destdir $DATADIRNEW --workers 20
```

#### Image pre-processing

Extract global image features using ResNet50. The pre-trained model is taken from (`https://download.pytorch.org/models/resnet50-0676ba61.pth`).

```
IMAGEDIRNEW=data/multi30k-images-test_2017_flickr
MODELFILE=models/resnet50-0676ba61.pth
FEATSDIR=data/multi30k-wmt18/task1-data/feats

python src/extract_image_feats_resnet50.py --image-folder $IMAGEDIRNEW --file-names $TEXT/image_splits/${TESTNAMENEW}.txt --batch-size 256 --model-file $MODELFILE --output-prefix $FEATSDIR/test
```

#### Text-only evaluation
```
fairseq-generate $DATADIRNEW --user-dir $FAIRSEQDIR --task translation --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --path $MODELSTEXTONLY/checkpoint_best.pt
```

#### Multimodal evaluation

Using ResNet50's average pooling layer output.

```
fairseq-generate $DATADIRNEW --user-dir $FAIRSEQDIR --task multimodal_translation --feats-path $FEATSDIR --feats-suffix '_resnet50-avgpool.npy' --feats-test-name $TESTNAMENEW --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --path $MODELSAVGPOOL/checkpoint_best.pt
```

Using ResNet50's res4f layer output.

```
fairseq-generate $DATADIRNEW --user-dir $FAIRSEQDIR --task multimodal_translation --feats-path $FEATSDIR --feats-suffix '_resnet50-res4frelu.npy' --feats-test-name $TESTNAMENEW --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --path $MODELSRES4FRELU/checkpoint_best.pt
```

## Description of models

### imgc_transformer


Variation 1 (dec+add): Add (transformed) multimodal features to the encoder output (i.e., decoder input).

Variation 2 (dec+concat): Concated multimodal features to the encoder output.

Variation 3 (enc+add): Add multimodal features to the word embeddings (i.e., encoder input).

Variation 4 (enc+concat): Concated multimodal features to sentence embeddings.

### imgatt_transformer

A visual-language attention layer is inserted into each layer of the decoder.
