#!/bin/bash

FV=$(pwd)
VT=$FV/vatex

if [ ! -d "${VATEX}" ]
then
	echo "NO VATEX"
	exit 1
fi

#format vatex folder
if [ ! -d "${VATEX}/raw" ]
then 
	mkdir $VATEX/raw
fi

if [ ! -d "${VATEX}/tok" ]
then
	mkdir $VATEX/tok
fi

if [ ! -d "${VATEX}/bpe" ]
then
	mkdir $VATEX/bpe
fi

TOK=$VATEX/tok
RAW=$VATEX/raw
BPE=$VATEX/bpe

#get raw captions
wget "https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json" -P $RAW
wget "https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json" -P $RAW
wget "https://eric-xw.github.io/vatex-website/data/vatex_public_test_english_v1.1.json" -P $RAW

#run preprocessing script on raw captions, tokenizing and saving to new files
cd $VATEX/scripts
python vatex_preprocess.py

cd $FV/subword-nmt

#10,000 merge operations are used [can be hyperparamaterized] 

#process is first completed in English
python ./learn_bpe.py -s 10000 < "${TOK}/train_tok.en" > "${TOK}/codes_en.bpe"
python ./apply_bpe.py -c "${TOK}/codes_en.bpe" < "${TOK}/train_tok.en" > "${BPE}/train.bpe10000.en

#process is repeated with Chinese
python ./learn_bpe.py -s 10000 < "${TOK}/train_tok.zh" > "${TOK}/codes_zh.bpe"
python ./apply_bpe.py -c "${TOK}/codes_zh.bpe" < "${TOK}/train_tok.zh" > "${BPE}/train.bpe10000.zh
