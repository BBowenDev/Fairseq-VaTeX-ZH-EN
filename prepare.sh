#!/bin/bash
pip install nltk
pip install jieba
FV=$(pwd)
echo "Checking External"
if [ ! -d "${FV}/external" ]
then
	mkdir $FV/external
	mkdir $FV/external/fairseq
	mkdir $FV/external/apex
	#install Fairseq
	cd $FV/external/fairseq
	git clone https://github.com/pytorch/fairseq
	cd fairseq
	git submodule update --init --recursive	
	pip install --editable ./
	echo "Installed Fairseq"
	#install Apex
	git clone https://github.com/NVIDIA/apex
	cd apex
	pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	echo "Installed Apex"
fi
echo "Checking subword-nmt"
if [ ! -d "${FV}/subword-nmt" ]
then		
	#install subword-nmt
	cd $FV
	git clone https://github.com/rsennrich/subword-nmt
	echo "Installed subword-nmt"
fi
