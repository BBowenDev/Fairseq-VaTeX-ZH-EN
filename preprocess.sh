#!/bin/bash

FV=$(pwd)
VT=$FV/vatex

#format vatex folder
echo "Formatting directories"
if [ ! -d "${VT}/raw" ]; then 
	mkdir $VT/raw
fi

if [ ! -d "${VT}/tok" ]; then
	mkdir $VT/tok
fi

if [ ! -d "${VT}/bpe" ]; then
	mkdir $VT/bpe
fi

TOK=$VT/tok
RAW=$VT/raw
BPE=$VT/bpe

#get raw captions
echo "Getting raw captions"
wget "https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json" -P $RAW
wget "https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json" -P $RAW
wget "https://eric-xw.github.io/vatex-website/data/vatex_public_test_english_v1.1.json" -P $RAW

#run preprocessing script on raw captions, tokenizing and saving to new files
echo "Tokenizing dataset"
cd $VT/scripts
python vatex_preprocess.py

#10,000 merge operations are used (can be hyperparamaterized) 
cd $FV/subword-nmt
echo "Learning BPE"
for TYPE in "train" "val" "test"; do
	for LANG in "en" "zh"; do 
		echo "--${TYPE}-${LANG}"
		
		INPUT="${TOK}/${TYPE}_tok.${LANG}"
		OUTPUT="${BPE}/${TYPE}.bpe10000.${LANG}"
		CODES="${TOK}/codes_${LANG}.bpe"
		VOCAB="${VT}/vocab"
		
		#no test file for ZH-- skip the BPE for that combination
		if [[ ! "$TYPE" == "test" && "$LANG" == "zh" ]]; then
			python ./subword_nmt/learn_bpe.py -s 10000 < $INPUT > $CODES
			python ./subword_nmt/apply_bpe.py -c $CODES --vocabulary $VOCAB < $INPUT > $OUTPUT
		fi
	done
done
