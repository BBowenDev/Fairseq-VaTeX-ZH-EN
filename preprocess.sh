#!/bin/bash

FV=$(pwd)
VT=$FV/vatex
SWNMT=$FV/subword-nmt

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

if [ ! -d "${VT}/vocab" ]; then
	mkdir $VT/vocab
fi

TOK=$VT/tok
RAW=$VT/raw
BPE=$VT/bpe
VOC=$VT/vocab

#get raw captions
echo "Getting raw captions"
wget "https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json" -P $RAW &
wget "https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json" -P $RAW &
wget "https://eric-xw.github.io/vatex-website/data/vatex_public_test_english_v1.1.json" -P $RAW &
wait

#run preprocessing script on raw captions, tokenizing and saving to new files
echo "Tokenizing dataset"
cd $VT/scripts
python vatex_preprocess.py

#10,000 merge operations are used (can be hyperparamaterized)
#learning and applying bpe are broken up so they can be parallelized
cd $SWNMT
echo "Learning BPE:"
for TYPE in "train" "test"; do #removed "val"
	for LANG in "en" "zh"; do 
		MERGES=10000
		INPUT="${TOK}/${TYPE}_tok.${LANG}"
		OUTPUT="${BPE}/${TYPE}.bpe${MERGES}.${LANG}"
		echo "trying ${OUTPUT}"
		CODES="${TOK}/codes_${LANG}.bpe"
		VOCAB="${VOC}/${TYPE}_vocab.${LANG}"
		echo "--${TYPE}-${LANG}"
		python $SWNMT/subword_nmt/learn_joint_bpe_and_vocab.py -s $MERGES -o $CODES --input $INPUT --write-vocabulary $VOCAB &
	done
done
wait

#once all BPE has been learned, it is applied
echo "Applying BPE:"
for TYPE in "train" "test"; do #removed "val"
	for LANG in "en" "zh"; do 
		INPUT="${TOK}/${TYPE}_tok.${LANG}"
		OUTPUT="${BPE}/${TYPE}.bpe${MERGES}.${LANG}"
		CODES="${TOK}/codes_${LANG}.bpe"
		VOCAB="${VOC}/${TYPE}_vocab.${LANG}"
		echo " trying ${TYPE}-${LANG}"
		#no test file for ZH-- skip the BPE for that combination
		if [[ "$TYPE" != "test" ]] && [[ "$LANG" != "zh" ]]; then
			echo "--${TYPE}-${LANG}"
			python $SWNMT/subword_nmt/apply_bpe.py -c $CODES --vocabulary $VOCAB < $INPUT > $OUTPUT &
		fi
	done
done
wait
