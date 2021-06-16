#!/bin/bash

FV= $(pwd)

if [ ! -d $FV/vatex/tok ] 
then
	mkdir vatex/tok
fi

TOK=$(pwd)/vatex/tok

if [ ! -d $FV/vatex/bpe ]
then
	mkdir vatex/bpe
fi

BPE=$(pwd)/vatex/bpe


cd vatex/scripts/
python vatex_preprocess.py

cd $FV/subword-nmt

#10000 merge operations are used for hyperparamaterization
python ./learn_bpe.py -s 10000 < $TOK/train_tok.en > codes.bpe

python ./apply_bpe.py -c codes.bpe < $TOK/train_tok.en > $BPE/train.bpe.en