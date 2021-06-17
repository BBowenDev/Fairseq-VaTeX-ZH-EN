#!/bin/bash

echo "Installing prerequisite packages"
pip install nltk 
pip install jieba

FV=$(pwd)

mkdir -p $FV/external/fairseq
mkdir -p $FV/external/apex

#install fairseq
echo "Installing Fairseq"
cd $FV/external/fairseq
git clone https://github/com/pytorch/fairseq
cd fairseq
git submodule update --init --recursive

if [ ! -d "${FV}/subowrd-nmt" ]
then 
	#install subword-nmt
	echo "Installing subword-nmt"
	cd $FV
	git clone https://github.com/rsennrich/subword-nmt
fi
