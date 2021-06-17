#!/bin/bash

pip install nltk
pip install jieba

FV=$(pwd)

mkdir -p $FV/external
mkdir -p $FV/external/fairseq
mkdir -p $FV/external/apex
	
#install Fairseq
cd $FV/external/fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
git submodule update --init --recursive	
pip install --editable ./

#install Apex
cd $FV/
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

if [ ! -d "${FV}/subword-nmt" ]
then
	#install subword-nmt
	cd $FV
	git clone https://github.com/rsennrich/subword-nmt
	cd subword-nmt
fi
