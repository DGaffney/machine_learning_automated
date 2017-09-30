import numpy as np
import datetime
from dateutil.parser import parse
from nltk.stem import PorterStemmer
import re
import csv
import os
import sys
import json

def label_type(y, y_type):
    if len(set(y)) == 2 and y_type in ["Categorical", "Float", "Integer"]:
        return "Binary"
    elif len(set(y)) <= 15 and y_type in ["Categorical", "Integer", "Phrase"]:
        return "Categorical"
    else:
        return "Ordinal"

def replaceiniter(it, predicate, replacement=None):
    for item in it:
        if predicate(item): yield replacement
        else: yield item

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #stemmer code from  - not the quite right place for this job but it'll be needed somewhere.
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r" \\\(  \\\(  \\\( ", " \(\(\( ", string)
    string = re.sub(r" \\\)  \\\)  \\\) ", " \)\)\) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "<URL/>", string)
    string = re.sub(r"www", " ", string)
    string = re.sub(r"com", " ", string)
    string = re.sub(r"org", " ", string)
    return string.strip().lower()

def read_json(filename):
    return json.loads(open(filename).read())

def read_csv(filename):
    rows = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    return rows

def cast_csv_given_manifest(rows, manifest):
    casted_dataset = []
    for row in rows:
        casted_row = []
        for i,val in enumerate(row):
            casted_row.append(cast_val(val, manifest['col_classes'][i]))
        casted_dataset.append(casted_row)
    return casted_dataset

def cast_val(value, directive):
    try:
        if directive == "Integer":
            return int(value)
        elif directive == "Float":
            return float(value)
        elif directive == "Time":
            if len(value) == 10 and sum(c.isdigit() for c in value) == 10:
                return datetime.datetime.fromtimestamp(int(value))
            elif len(value) == 13 and sum(c.isdigit() for c in value) == 13:
                return datetime.datetime.fromtimestamp(int(value)/1000)
            else:
                return parse(value, fuzzy=True)
        elif directive == "Text" or directive == "Phrase":
            return [PorterStemmer().stem(word) for word in clean_str(value).split(" ")]
        elif directive == "Categorical":
            return value
    except:
        return None


def parse(data_filename, manifest_filename):
    rows = read_csv(data_filename)
    manifest = read_json(manifest_filename)
    return convert_text_fields_to_data(cast_csv_given_manifest(rows, manifest), manifest), manifest

def convert_text_fields_to_data(casted_dataset, manifest):
    transposed = np.array(casted_dataset).transpose().tolist()
    detexted = []
    labels = []
    conversion_pipeline = {}
    for i,col in enumerate(transposed):
        if i == int(manifest['prediction_column']):
            labels = col
        elif manifest['col_classes'][i] == "Phrase" or manifest['col_classes'][i] == "Text":
            #future feature is to do word-after-word approach vis-a-vis RNNs/CNNs instead of simple counts
            unique_terms = list(set([item for sublist in col for item in sublist]))
            conversion_pipeline[i] = {"unique_terms": unique_terms}
            counteds = []
            for term in unique_terms:
                counted = []
                for row in col:
                    counted.append(row.count(term))
                detexted.append(counted)
        elif manifest['col_classes'][i] == "Categorical":
            unique_vals = list(set(col))
            conversion_pipeline[i] = {"unique_terms": unique_terms}
            newcol = []
            for val in col:
                newcol.append(unique_vals.index(val))
            detexted.append(newcol)
        else:
            average = np.mean([c for c in col if c != None])
            conversion_pipeline[i] = {"average": average}
            detexted.append(list(replaceiniter(col, lambda x: x==None, average)))
    return np.array(detexted).transpose().tolist(), labels, conversion_pipeline