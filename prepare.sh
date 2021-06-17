#!/bin/bash

echo "Installing Prerequisites"

pip install nltk
pip install jieba

FV=$(pwd)

if [ ! -d "${FV}/external" ]
then 
	#create missing directories
	mkdir $FV/external
	
	#install fairseq
	cd $FV/external
	git clone https://github.com/pytorch/fairseq
	cd fairseq
	git submodule update --init --recursive
	pip install --editable ./
fi

if [ ! -d "${FV}/subword-nmt" ]
then
	#install subword-nmt
	cd $FV
	git clone https://github.com/rsennrich/subword-nmt
fi
