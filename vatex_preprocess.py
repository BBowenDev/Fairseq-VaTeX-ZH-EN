import json
import jieba
import nltk
from nltk.tokenize import word_tokenize
import os

#if nltk has never been used, run line:
nltk.download("punkt")

raw_path = str(os.path.abspath("..\\raw\\") + "\\")
tok_path = str(os.path.abspath("..\\tok\\") + "\\")

jsons = ["vatex_training_v1.0", "vatex_validation_v1.0", "vatex_public_test_english_v1.1"]
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
            vtx_dict["en"] += raw_dict["enCap"][-5:]
        else: 
            vtx_dict["en"] = False
        
        if "chCap" in raw_dict.keys():
            vtx_dict["zh"] += raw_dict["chCap"][-5:]
        else: 
            vtx_dict["zh"] = False
    
    formatted[out_files[num]] = vtx_dict


#tokenize, remove cases, and save to tok
print("Tokenizing:")
for file in out_files:
    for lang in langs:
        if not formatted[file][lang] == False:
            with open(tok_path + file + "_tok" + "." + lang, "w", encoding="utf-8") as f:
                print("--", file + "." + lang)
                
                if lang == "en":
                    for line in formatted[file][lang]:
                        words = word_tokenize(line.lower())
                        words = " ".join(words)
                        f.write(words + "\n") 
                
                elif lang == "zh":
                    jieba.initialize()
                    jieba.setLogLevel(20)
                    for line in formatted[file][lang]:
                        words = jieba.lcut(line, cut_all=True)
                        words = " ".join(words)
                        f.write(words + "\n") 