# -*- coding: utf-8 -*-
# !/usr/bin/python3
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
# REVIEWER wenchen:  document the file 
"""
ner
author: Shuhao Liu
description: use spacy to implement ner model
"""

from __future__ import unicode_literals, print_function
from data_reader import read_inception_json,transfer_form,write_in_newdata,clean_data,clean_data_tolist,read_in_old,json_to_trainlist
import plac
import random
import pickle
import torch
from pathlib import Path
import spacy
import json
import math
import pandas as pd
from spacy.util import minibatch, compounding
import os, sys
from spacy.gold import docs_to_json
from spacy.gold import GoldParse
from spacy.scorer import Scorer

# REVIEWER wenchen:  delete this
#import thinc.neural.gpu_ops
# get the trainning data from json by using method in data_reader file


class NLPModel(object):
    def __init__(self, model, output_dir,n_iter,data):
        # REVIEWER wenchen:  refer this https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder_impl.py
        '''
        initialize the NlpModel
        
        output_dir is the direction of file used to save training model
        n_iter is the running time for training
        DATA is the tupple of three data sets already annotated, in each dataset the form is like
        dATA = [
        ("Who is Slhaka Khan?", {"entities": [(7, 17, "PERSON")]}),
        ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
        the first data set is train_data
        the second data set is dev data set
]
        '''
        self.model = model
        self.output_dir = output_dir
        self.n_iter = n_iter
        self.train_data = data[0]
        self.dev_data = data[1]
        if self.model is not None:
            #self.model = spacy.load(self.model)  # load existing spaCy model
            print("Loaded model '%s'" % model)
        else:
            self.model = spacy.blank("en")  # create blank Language class
            print("Created blank 'en' model")
            doc = self.model('harry is apple')
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
        # REVIEWER wenchen:  move (spacy) model init here TODO 1

        
    def train(self):
        '''
        use train data set to test the self.model
        '''

        nlp = self.model
        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner, last=True)
        # otherwise, get it so we can add labels
        else:
            ner = nlp.get_pipe("ner")
        
        
        # add labels
        for _, annotations in self.train_data :
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):  # only train NER
            # reset and initialize the weights randomly – but only if we're
            # training a new model
            nlp.begin_training()
            for itn in range(self.n_iter):
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(self.train_data, 124)#compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        losses=losses,
                    )
                print("Losses", losses)
        self.model = nlp
        
        
    def update(self,data):
        '''
        dataset[0] is the train_data
        dataset[1] is the dev_data
        '''
        self.train_data = data[0]
        self.dev_data = data[1]
    
    
    
        
    def save_model(self):  # REVIEWER wenchen: saved model add to ignore not push to git
    # save model to output directory
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            if not self.output_dir.exists():
                self.output_dir.mkdir()
            self.model.to_disk(self.output_dir)
            print("Saved model to", self.output_dir)
            
    def evaluation(self):
        '''
        Update the evaluation scores from a single Doc / GoldParse pair./
        output is {usa:unlabelled dependency score,las:labelled dependency score,ents_p:Name entity accuracy,
        ents_r:Name entity recall,ents_f:Name entity F-score,ents_per_type:score per entity label}
        '''
        if self.model is None:
            print("no model")
        else:
            train = Scorer()
            dev = Scorer()
            for input_, annot in self.train_data:
                doc_gold_text = self.model.make_doc(input_)
                gold = GoldParse(doc_gold_text ,entities=annot['entities'])
                pred_value = self.model(input_)
                train.score(pred_value , gold)
            for input_, annot in self.dev_data:
                doc_gold_text = self.model.make_doc(input_)
                gold = GoldParse(doc_gold_text ,entities=annot['entities'])
                pred_value = self.model(input_)
                dev.score(pred_value , gold)
            t = train.scores
            data = {'precision':[t['ents_p']],'recall':[t['ents_r']],'f-score':[t['ents_f']],'support':[0]}
            df = pd.DataFrame(data)
            df.index = pd.Series(['ave/total'])
            total = 0
            for i in t['ents_per_type'].keys():
                if len(t['ents_per_type'][i]) != 0 :
                    support = train.ner_per_ents[i].tp + train.ner_per_ents[i].fp + train.ner_per_ents[i].fn
                    total = total+ support
                    df.loc[str(i)] =[str(t['ents_per_type'][i]['p'])[:5],str(t['ents_per_type'][i]['r'])[:5],str(t['ents_per_type'][i]['f'])[:5],support]
            df.loc['ave/total','support'] = total
            with open('data/evaluation.txt', 'w') as fo:
                fo.write('train_data\n')
                fo.write(df.__repr__())
                fo.close()
                
            
            d = dev.scores
            dev_data = {'precision':d['ents_p'],'recall':d['ents_r'],'f-score':d['ents_f'],'support':[0]}
            ff = pd.DataFrame(dev_data)
            ff.index = pd.Series(['ave/total'])
            total = 0
            for i in d['ents_per_type'].keys():
                if len(d['ents_per_type'][i]) != 0 :
                    support = dev.ner_per_ents[i].tp + dev.ner_per_ents[i].fp + dev.ner_per_ents[i].fn
                    total = total+ support
                    ff.loc[str(i)] =[str(d['ents_per_type'][i]['p'])[:5],str(d['ents_per_type'][i]['r'])[:5],str(d['ents_per_type'][i]['f'])[:5],support]
            ff.loc['ave/total','support'] = total
            with open('data/evaluation.txt', 'a') as fo:
                fo.write('\ndev_data\n')
                fo.write(ff.__repr__())
                fo.close()
            

            #return(train.score(pred_value , gold),dev.score(pred_value , gold))
            #print ("train_data evaluation:"+ str(train.scores))     
            #print ("dev_data evaluation:"+ str(dev.scores))    
        
    # REVIEWER wenchen: separate input_data with predict function 
    def predict(self, data):
        doc = self.model(data)
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
            
# test the model
                        
def shuffle_data(data):
        '''
        shuffle the data, and split it to training data set(three forths of dataset) and test data set(last one forth)
        '''
        random.shuffle(data)
        train_data = data[:8*(math.floor(len(data)/10))]
        dev_data = data[8*(math.floor(len(data)/10)):9*(math.floor(len(data)/10))]
        test_data = data[9*(math.floor(len(data)/10)):]
        return(train_data,dev_data,test_data)
    
def load_model(load_dir):
        if load_dir is not None:
        # test the saved model
            print("Loading from", load_dir)
            model = spacy.load(load_dir)
            return model
        
def evaluation(model, data):
    print("--------test evaluation-------")
    '''
    Update the evaluation scores from a single Doc / GoldParse pair./
    output is {usa:unlabelled dependency score,las:labelled dependency score,ents_p:Name entity accuracy,
    ents_r:Name entity recall,ents_f:Name entity F-score,ents_per_type:score per entity label}
    '''
    if model is None:
        print("no model")
    else:
        evaluation = Scorer()
        for input_, annot in data:
            doc_gold_text = model.make_doc(input_)
            gold = GoldParse(doc_gold_text ,entities=annot['entities'])
            pred_value = model(input_)
            evaluation.score(pred_value , gold)
        result = evaluation.scores
        test_data = {'precision':result['ents_p'],'recall':result['ents_r'],'f-score':result['ents_f'],'support':[0]}
        train_dataframe = pd.DataFrame(test_data)
        train_dataframe.index = pd.Series(['ave/total'])
        total = 0
        for i in result['ents_per_type'].keys():
            if len(result['ents_per_type'][i]) != 0 :
                support = evaluation.ner_per_ents[i].tp + evaluation.ner_per_ents[i].fp + evaluation.ner_per_ents[i].fn
                total = total+ support
                train_dataframe.loc[str(i)] =[str(result['ents_per_type'][i]['p'])[:5],str(result['ents_per_type'][i]['r'])[:5],str(result['ents_per_type'][i]['f'])[:5],support]
        train_dataframe.loc['ave/total','support'] = total
        with open('data/evaluation.txt', 'a') as fo:
            fo.write('\ntest_data\n')
            fo.write(train_dataframe.__repr__())    
            fo.close() 


def save_unclean_data(path,data):
    nerd_dic = clean_data(data)
    with open(path, 'w') as f:
        json.dump(nerd_dic, f)
        f.close()



def read_in_new(path):
    '''
    read in new data 
    '''
    write_in_path = 'data/my_dict.json'
    read_in_path = 'data/my_dict.json'
    data = json_to_trainlist(path)
    write_in_newdata(read_in_path,data)
    database = read_in_old(read_in_path)
    with open('data/transform_data.pkl', 'wb') as f:
       pickle.dump(database, f)
    data = shuffle_data(database)
    return data

def train_data(model,path,data):
    nlp = NLPModel(model,path,100,data[0:2])
    nlp.train()
    nlp.save_model()
    nlp.evaluation()
    evaluation(nlp.model,data[2])
    


def Evaluate_Accurate(model,testdata):
    if str(type(model)) =="<class 'NoneType'>":
        print("none model")
    else:
        sumNum = 0
        trueNum = 0
        for text, dic in testdata :
            doc = model(text)
            Original_Entities = []
            for i in dic["entities"]:
                Original_Entities.append(i[-1])
            #print([(ent.label_) for ent in doc.ents])
            #print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            #print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
            #print("_____________"）
            predict_Entities = [(ent.label_) for ent in doc.ents]
            if Original_Entities == predict_Entities:
                trueNum += 1
            sumNum +=1
        print(trueNum/sumNum) 
        
        
def read_in_new(path):
    '''
    read in new data 
    '''
    database = read_json_file(path)
    (train,dev,test) = shuffle_data(database)
    
    train_path = 'data/bert_train.pkl'
    test_path = 'data/bert_test.pkl'
    valid_path = 'data/bert_valid.pkl'
    with open(train_path, 'wb') as f:
       pickle.dump(train, f)
    with open(test_path, 'wb') as f:
       pickle.dump(test, f)
    with open(valid_path, 'wb') as f:
       pickle.dump(dev, f)
    
    return (train,dev,test)

def read_json_file(path):
    with open(path) as f:
        my_dict = json.load(f)
    l = []
    for i in my_dict.keys():
        l.append((i,my_dict[i]))
    return l



if __name__ == "__main__":
    spacy.require_gpu()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth',400)
    nerd_file ="data/wencai_ned.json"
    nerd_path ="data/ner_clean_2.json"
    dirName = 'output.json'
    dic_path = 'data/my_dict.json'  # the path saved data in dictionary
    # read in the unclean data and save it in dic_path
    data = json_to_trainlist(nerd_file)
    save_unclean_data(dic_path,data)
    nerd_path ="data/my_dict.json"
    data = read_in_new(nerd_path)
    # data already shuffled in read_in_new function
    train_data(None,dirName,data)
    #Evaluate_Accurate(load_model(dirName),data[2])
    x = input("test sentence:")
    while x != 'end':
        nlp =load_model(dirName)
        doc = nlp(x)
        print(doc)
        #print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
        x = input("test sentence:")
    