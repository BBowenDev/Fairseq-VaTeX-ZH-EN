#!/bin/bash

pip install nltk
pip install jieba

FV=$(pwd)

cd $FV
if [ ! -d $FV/fairseq ]
then
	git clone https://github.com/pytorch/fairseq
	cd fairseq
	git submodule update --init --recursive	
	pip install --editable ./
fi

if [ ! -d $FV/subword-nmt ]
then
	cd $FV
	git clone https://github.com/rsennrich/subword-nmt
	cd subword-nmt
fi