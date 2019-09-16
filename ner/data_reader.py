# -*-: coding:utf-8 -*-
# Copyright 2019 Hithink Flush Information Network Co Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
author: Shuhao Liu
contact: harrymj19961007@gmail.com

nerd annotation:
corpus: text corpus with all the sentences in one file
NamedEntity:
    "begin" : begin index in the corpus
    "end" : end index in the corpus
    "value" :  named entity tag
    "identifier" : named entity linking url

identifier website:
    1. dbpedia
    2. wikipedia
    3. investopedia
    4. yahoo finance( for ORG and company)

"""
import json
def read_inception_json(file_path):
    '''
    read the json file from it's path and return the (nameEnity,corpus)
    '''
    with open(file_path, encoding='utf-8-sig') as json_file:
        json_data = json.load(json_file)
        corpus = json_data["_referenced_fss"]["12"]["sofaString"]
        namedEntity = json_data["_views"]["_InitialView"]["NamedEntity"]
    return (namedEntity,corpus)

def transfer_form(namedEntity,corpus):
    """
    transfer (nameEnitity,corpus) to the train list that 
    accord to the spacy input form
    """
    listCorpus = []
    j = 0
    for i in range(len(corpus)) :
        if corpus[i] == '\n':
            listCorpus.append([j,i])
            j = i+1
    train_list = []
    for [a,b] in listCorpus:
        token = (corpus[a:b])
        entitylist = []
        while namedEntity[0]["end"] <= b :
            entitylist.append(tuple((namedEntity[0]["begin"]-a,namedEntity[0]["end"]-a,namedEntity[0]["value"])))
            namedEntity.pop(0)
            if len(namedEntity) == 0:
                break
        train_list.append((token,{"entities":entitylist}))
    return(train_list)

def json_to_trainlist(file_path):
    (a,b) = read_inception_json(file_path)
    return transfer_form(a,b)

def clean_data(data):
    '''
    clean the data with same text
    and store data into a dictionary
    '''
    nerd_dic = {}
    for text,annot in data:
        nerd_dic[text] = annot
    return (nerd_dic)

def clean_data_tolist(data):
    '''
    clean the data wieh same text
    and store data into a list
    '''
    nerd_dic = {}
    for text,annot in data:
        nerd_dic[text] = annot
    l = []
    for i in nerd_dic.keys():
        l.append((i,nerd_dic[i]))
    return l



# clean the new data and put it in to the txt
# use this function to filter data and used for Inception
def filter_new_data(file_path):
    (a,b) = read_inception_json(file_path)
    list_corpus = b.split("\n")
    dict_corpus = {}
    for i in list_corpus:
        dict_corpus[i] = "new"
    with open('data/my_dict.json') as f:
        my_dict = json.load(f)
    f = open('data/input_file', 'w')
    for text in dict_corpus.keys() :
        if text not in my_dict.keys():
            f.write(text+'\n')
    f.close()
def write_in_newdata(read_in_path,data):
    '''
    use dictionary to combine old and new data
    then save it in to a file
    '''
    with open(read_in_path) as f:
        my_dict = json.load(f)
    for text,annot in data:
        my_dict[text] = annot
    with open(read_in_path, 'w') as f:
        json.dump(my_dict, f)
        
def read_in_old(path):
    '''
    read the privious data saved in the path
    the output is all of the (entity,corpus) that can be put in the spacy
    '''
    with open(path) as f:
        my_dict = json.load(f)
    l = []
    for i in my_dict.keys():
        l.append((i,my_dict[i]))
    return l
    



