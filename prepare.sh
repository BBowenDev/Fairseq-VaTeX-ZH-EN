#!/bin/bash

echo "Installing Prerequisites"

pip install nltk
pip install jieba
pip install sacremoses

FV=$(pwd)

if [ ! -d "${FV}/models" ]; then
	mkdir $FV/models
fi

if [ ! -d "${FV}/vatex" ]; then
	#create vatex folders
	mkdir $FV/vatex
	mkdir $FV/vatex/scripts
fi

mv vatex_preprocess.py $FV/vatex/scripts

#check CUDA installation/version
#CV = $(nvcc --version)
#if [ "${CV}" != *"release 10.2"* ]; then
#	apt-get install cuda-10-2 &
#	wait
#fi

if [ ! -d "${FV}/external" ]; then 
	#create missing directories
	mkdir $FV/external
	
	#install fairseq
	echo "Installing Fairseq"
	cd $FV/external
	git clone https://github.com/pytorch/fairseq
	cd fairseq
	git submodule update --init --recursive
	pip install fairseq &
	wait
	
	#install apex
	echo "Installing Apex"
	cd $FV/external
	git clone https://github.com/NVIDIA/apex
	cd apex
	python setup.py install --cuda_ext --cpp_ext --pyprof
	pip install apex &
	wait
fi

if [ ! -d "${FV}/subword-nmt" ]; then
	#install subword-nmt
	cd $FV
	git clone https://github.com/rsennrich/subword-nmt
fi
