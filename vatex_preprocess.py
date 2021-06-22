import argparse
import json
import jieba
import nltk
from nltk.tokenize import word_tokenize
import os

parser = argparse.ArgumentParser(description="A preprocessing script to tokenize the VaTeX dataset.")
parser.add_argument("-f", "--full", dest="full", default="True", 
                    help="True if using the full dataset, False if only using parallel translations")
parser.add_argument("-t", "--test_size", dest="test_size", type=int, default=10000, 
                    help="Number of sentenced removed from validation set to be used for training. Default 10,000, max 29,999")
args = parser.parse_args()

if args.test_size > 29999:
    raise argparse.ArgumentError("Maximum test size of 29,9999")
    quit()

#if nltk has never been used, run line:
nltk.download("punkt")

raw_path = str(os.path.abspath("../raw/") + "/")
tok_path = str(os.path.abspath("../tok/") + "/")

jsons = ["vatex_training_v1.0", "vatex_validation_v1.0"]
out_files = ["train", "val", "test"]
langs = ["en", "zh"]

#format listed sentences from raw json data             
formatted = {}

print("Reading:")
for num, data_file in enumerate(jsons):        
    with open(raw_path + data_file + ".json") as f:
        print(data_file + ".json" + " opening")
        data = json.load(f)
     
    vtx_dict = {"en": [], "zh": []}
    
    print("--", data_file)
    for raw_dict in data:
        if "enCap" in raw_dict.keys():
            if args.full == "True": #if using the full dataset, don't truncate
                vtx_dict["en"] += raw_dict["enCap"]
            else: #otherwise, truncate dataset to parallel captions only
                vtx_dict["en"] += raw_dict["enCap"][-5:]
        
        if "chCap" in raw_dict.keys():
            if args.full == "True": #if using the full dataset, don't truncate
                vtx_dict["zh"] += raw_dict["chCap"]
            else: #otherwise, truncate dataset to parallel captions only
                vtx_dict["zh"] += raw_dict["chCap"][-5:]
                
    formatted[out_files[num]] = vtx_dict

#format val set to test set
formatted["test"] = {"en": [], "zh": []}
formatted["test"]["en"] = formatted["val"]["en"][0:args.test_size]
formatted["val"]["en"] = formatted["val"]["en"][args.test_size:]
formatted["test"]["zh"] = formatted["val"]["zh"][0:args.test_size]
formatted["val"]["zh"] = formatted["val"]["zh"][args.test_size:]
    
#tokenize, remove cases, and save to tok
print("Tokenizing:")
jieba.initialize()
jieba.setLogLevel(20)

for file in out_files:
    for lang in langs:

            with open(tok_path + file + "_tok" + "." + lang, "w", encoding="utf-8") as f:
                print("--", file + "." + lang)
                
                if lang == "en":       
                    for line in formatted[file][lang]:
                        words = word_tokenize(line.lower()) #tokenize English with nltk
                        words = " ".join(words)
                        f.write(words + "\n") 
                
                elif lang == "zh":
                    for line in formatted[file][lang]:
                        words = jieba.lcut(line, cut_all=True) #tokenize Chinese with jieba
                        words = " ".join(words)
                        f.write(words + "\n") 