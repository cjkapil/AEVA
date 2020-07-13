import load_data
import features
import models
import pickle
import eel
eel.init('web')
def extract_features(essay, f):
    vectorizer = pickle.load(f)
    train_features, train_bog = pickle.load(f)
    clean_essay = []
    clean_essay.append(features.essay_to_words(essay[0]))
    test_bog = vectorizer.transform(clean_essay).toarray()
    test_features = features.syn_features(clean_essay)
    test_features = test_features.values
    #print(train_bog.shape, test_bog.shape)
    #print(test_features)
    return train_features, test_features, train_bog, test_bog    
    
def main():
    infile1=open("/home/chahal/Desktop/ocr/vision/IO_files/data.txt")
    infile2=open("/home/chahal/Desktop/ocr/vision/IO_files/setnum.txt")
    
    text=infile1.read()
    in_essay=[]
    in_essay.append(text)
    #print(in_essay)
    essay_set=int(infile2.read())#set1=0 set2.1=1 set 2.2=2 set 3=3...
    

    files=['/home/chahal/Desktop/AEVAtest/TA_model/set1.pkl', '/home/chahal/Desktop/AEVAtest/TA_model/set2_1.pkl', '/home/chahal/Desktop/AEVAtest/TA_model/set2_2.pkl', '/home/chahal/Desktop/AEVAtest/TA_model/set3.pkl', '/home/chahal/Desktop/AEVAtest/TA_model/set4.pkl', '/home/chahal/Desktop/AEVAtest/TA_model/set5.pkl', '/home/chahal/Desktop/AEVAtest/TA_model/set6.pkl', '/home/chahal/Desktop/AEVAtest/TA_model/set7.pkl', '/home/chahal/Desktop/AEVAtest/TA_model/set8.pkl']  

    f = open(files[essay_set], 'rb')
    train_features, test_features, train_bog, test_bog = extract_features(in_essay,f)
    meta_test, predicted_scores = models.predictor(train_features, test_features, train_bog, test_bog, f)
    score = round(predicted_scores[0])
    eel.reply(score)
    print(score)
    return score


if __name__=="__main__":
    main()
