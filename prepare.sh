#!/bin/bash

echo "Installing Prerequisites"

pip install nltk
pip install jieba

FV=$(pwd)

if [ ! -d "${FV}/vatex" ]
then
	#create vatex folders
	mkdir $FV/vatex
	mkdir $FV/vatex/scripts
fi

mv vatex_preprocess.py $FV/vatex/scripts

if [ ! -d "${FV}/external" ]
then 
	#create missing directories
	mkdir $FV/external
	
	#install fairseq
	echo "Installing Fairseq"
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
