import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import scipy as sc
import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.data
import enchant
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
    
def essay_to_words(raw_essay):

##    1. remove punctuations and numbers
    letters_only = re.sub("[^a-zA-Z]", " ", raw_essay)
    
##    2.Convert to lower case
    essays = letters_only.lower()

##    3.Word tokenize
    words = word_tokenize(essays)


##    4.Remove stop words, searching a set is faster
    stop_words = set(stopwords.words( "english"))
    filtered_words = [w for w in words if w not in stop_words]

##    5. Lemmatization
    meaningful_words=[]
    tag=nltk.pos_tag(filtered_words)
    for i,j in tag:
        if j=="NN":
            j='n'
        if j=="JJ":
            j='j'
        if j=="RB":
            j='r'
        if j=="VB":
            j='v'
        else:
            j='n'
        l=WordNetLemmatizer()
        word=l.lemmatize(i,j)
        meaningful_words.append(word)
##  6. join the words into one string
    final_essay = ' '.join(meaningful_words)

    return final_essay
        
        




def bag_of_words(essay_list, test_list,f):

    train=[]
    test=[]
    
    for item in essay_list:
        clean_essay=essay_to_words(item)
        train.append(clean_essay)

    for item in test_list:
        clean_essay=essay_to_words(item)
        test.append(clean_essay)

    vectorizer = TfidfVectorizer(ngram_range=(1,4),max_df=.9, min_df=0.1, max_features=200000)
    vectorizer.fit(train)
    pickle.dump(vectorizer,f)
    train_features = vectorizer.transform(train)
    test_features = vectorizer.transform(test)

   # print(vectorizer.get_feature_names())

    train_features = train_features.toarray()
    test_features = test_features.toarray()
    print("inside bow")
    return train_features, test_features


def get_word_count(essay,count):
    return [len(essay)]

def get_spelling_mistakes(essay, count):
    d = enchant.Dict("en_US")
    sp_miss = []
    for i in essay:
        if d.check(i):
            m=0
        else :
            sp_miss.append(i)
    return [len(sp_miss)/count]
    
def get_distinct_words_count(essay, count):
    return [len(set(w.lower for w in essay))/count]

def get_avg_sent_length(essay):
    sent = sent_tokenize(essay)
    sent_list = [len(item.split()) for item in sent]
    return len(sent_list), sum(sent_list)/len(sent_list)


def get_tag_count(essay, count) :
    tag_list=[]
    noun=0
    adjective=0
    adverb=0
    verb = 0
    deter = 0
    pronoun = 0
    wh = 0
    tag=nltk.pos_tag(essay)
    for i, j in tag:
        if 'NN' in j:
            noun+=1
        if 'JJ' in j:
            adjective+=1
        if 'RB' in j:
            adverb+=1
        if 'VB' in j:
            verb+=1
        if 'DT' in j:
            deter+=1
        if 'PR' in j:
            pronoun+=1
        if 'W' in j:
            wh+=1
    return [noun/count, verb/count, adjective/count, adverb/count, deter/count, pronoun/count, wh/count]


    
    



def extract_features(essays, feature_functions):

    #create list of features for each essay
    features = []
    for es in essays:
        flist = []
        words = re.sub("[^a-zA-Z ]", " ", es)
        token = word_tokenize(words)
        count, avg =get_avg_sent_length(es)
        for f in feature_functions:
           flist.extend(f(token, count))
        flist.extend([count, avg])
        features.append(flist)
    return features

def syn_features(essays):

    essay_list = []
    feature_functions = [get_word_count, get_spelling_mistakes, get_distinct_words_count, get_tag_count]
    
    features = extract_features(essays, feature_functions)
    df = pd.DataFrame(features)
    df.columns = ['word count', 'spelling mistakes', 'distinct words count', 'noun', 'verb', 'adj', 'adverb', 'determinator', 'pronoun', 'wh words', 'sentence count','avg sent len'] 
    return df

if __name__ =="__main__":
    print("start\n")

    essays = ["Dear sir. This is a fine day.", "how are you", "we write this as some normal human, but are we really humans"]
    bog = bag_of_words(essays)
    print(essays)
    print("\n bog\n")
    print(bog)
    print("helllo")
    features = syn_features(essays)
    print("syntatic features")
    print(features)
    
