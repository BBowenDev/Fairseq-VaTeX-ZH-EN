#!/bin/bash

FV= $(pwd)
VATEX=$FV/vatex

#format vatex folder
if [ ! -d $VATEX/raw ] 
then 
	mkdir $VATEX/raw
fi

if [ ! -d $VATEX/tok ] 
then
	mkdir $VATEX/tok
fi

if [ ! -d $VATEX/bpe ]
then
	mkdir $VATEX/bpe
fi

TOK=$VATEX/tok
RAW=$VATEX/raw
BPE=$VATEX/bpe

#get raw vatex captions
wget https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json $RAW
wget https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json $RAW
wget https://eric-xw.github.io/vatex-website/data/vatex_public_test_english_v1.1.json $RAW

#run preprocessing script on raw captions, tokenizing and saving to new files.
cd $VATEX/scripts/
python vatex_preprocess.py

cd $FV/subword-nmt

#10000 merge operations are used for hyperparamaterization
python ./learn_bpe.py -s 10000 < $TOK/train_tok.en > codes.bpe
python ./apply_bpe.py -c codes.bpe < $TOK/train_tok.en > $BPE/train.bpe.en