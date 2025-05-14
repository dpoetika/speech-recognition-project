import pandas as pd
import csv 
import os

tsv_path = "C:/Users/nebul/Desktop/python/audio/speech-recognition-project/src/isai.tsv"


basliklar = ["client_id","path","sentence_id","sentence","sentence_domain","up_votes","down_votes","age","gender","accents","variant","locale","segment"]
folder = "C:/Users/nebul/Downloads/ISSAI_TSC_218/ISSAI_TSC_218"
alters = ["train","test","dev"]

with open(tsv_path, 'a') as tsvfile:
    writer = csv.DictWriter(tsvfile, fieldnames=basliklar,delimiter='\t',lineterminator="\r")
    writer.writeheader()

    id = 1
    while True:
        for i in alters:
            finded = os.path.join(folder,f"{i}/{id}.txt")
            if os.path.exists(finded):
                break

        if not os.path.exists(finded):

            print(id)
            id += 1
            continue

        
        dict_path = os.path.join(folder,f"{i}/{id}.wav")
        dict_sentence = open(finded).readline().rstrip()
    
        writer.writerow({"path":f"{dict_path}","sentence":f"{dict_sentence}"})
        
        
        id += 1

