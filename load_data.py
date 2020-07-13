import re

def add_essay_training(data, essay_set, essay, score):
    if essay_set not in data:
        data[essay_set] = {"essay":[],"score":[]}
    data[essay_set]["essay"].append(essay)
    data[essay_set]["score"].append(score)

def add_essay_test(data, essay_set, essay, prediction_id):
    if essay_set not in data:
        data[essay_set] = {"essay":[], "prediction_id":[]}
    data[essay_set]["essay"].append(essay)
    data[essay_set]["prediction_id"].append(prediction_id)

def read_training_data(training_file):
    f = open(training_file,"r",encoding='mac_roman')
    f.readline()


    training_data = {}
    for row in f:
        row = row.strip().split("\t")
        
        essay_set = row[2]
        essay = row[3]
        domain1_score = float(row[7])
        if essay_set == "2":
            essay_set = "2_1"
        add_essay_training(training_data, essay_set, essay, domain1_score)
        
        if essay_set == "2_1":
            essay_set = "2_2"
            domain2_score = float(row[10])
            add_essay_training(training_data, essay_set, essay, domain2_score)
    
    return training_data

def read_test_data(test_file):
    
    f = open(test_file,"r",encoding='mac_roman')
    f.readline()
    
    test_data = {}
    for row in f:
        row = row.strip().split("\t")
        essay_set = row[2]
        essay = row[3]
        
        domain1_predictionid = float(row[4])
        if essay_set == "2": 
            domain2_predictionid = float(row[5])
            add_essay_test(test_data, "2_1", essay, domain1_predictionid)
            add_essay_test(test_data, "2_2", essay, domain2_predictionid)
        else:
            add_essay_test(test_data, essay_set, essay, domain1_predictionid)
    return test_data

def load():
    print("Reading Training Data")
    training = read_training_data("data.tsv")
    print("Reading Validation Data")
    test = read_test_data("test_data.tsv")
    return training, test
