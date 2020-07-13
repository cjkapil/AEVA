import os, io
from google.cloud import vision
from google.cloud.vision import types
import argparse
import sys
import cv2
import editdistance
import numpy as np
import tensorflow as tf
import pandas as pd


from OCR_DataLoader import Batch, DataLoader, FilePaths
from OCR_Preprocess import preprocessor, wer
from OCR_Model import DecoderType, Model
import index_main
import chatbot
import eel

eel.init('web')


def train(model, loader):
    """ Train the neural network """
    epoch = 0  # Number of training epochs since start
    bestCharErrorRate = float('inf')  # Best valdiation character error rate
    noImprovementSince = 0  # Number of epochs no improvement of character error rate occured
    earlyStopping = 25  # Stop training after this number of epochs without improvement
    batchNum = 0

    totalEpoch = len(loader.trainSamples)//Model.batchSize 

    while True:
        epoch += 1
        print('Epoch:', epoch, '/', totalEpoch)

        # Train
        print('Train neural network')
        loader.trainSet()
        while loader.hasNext():
            batchNum += 1
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch, batchNum)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # Validate
        charErrorRate, addressAccuracy, wordErrorRate = validate(model, loader)
        cer_summary = tf.Summary(value=[tf.Summary.Value(
            tag='charErrorRate', simple_value=charErrorRate)])  # Tensorboard: Track charErrorRate
        # Tensorboard: Add cer_summary to writer
        model.writer.add_summary(cer_summary, epoch)
        address_summary = tf.Summary(value=[tf.Summary.Value(
            tag='addressAccuracy', simple_value=addressAccuracy)])  # Tensorboard: Track addressAccuracy
        # Tensorboard: Add address_summary to writer
        model.writer.add_summary(address_summary, epoch)
        wer_summary = tf.Summary(value=[tf.Summary.Value(
            tag='wordErrorRate', simple_value=wordErrorRate)])  # Tensorboard: Track wordErrorRate
        # Tensorboard: Add wer_summary to writer
        model.writer.add_summary(wer_summary, epoch)

        # If best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # Stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' %
                  earlyStopping)
            break


def validate(model, loader):
    """ Validate neural network """
    print('Validate neural network')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0

    totalCER = []
    totalWER = []
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        recognized = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            ## editdistance
            currCER = dist/max(len(recognized[i]), len(batch.gtTexts[i]))
            totalCER.append(currCER)

            currWER = wer(recognized[i].split(), batch.gtTexts[i].split())
            totalWER.append(currWER)

            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' +
                  batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # Print validation result
    charErrorRate = sum(totalCER)/len(totalCER)
    addressAccuracy = numWordOK / numWordTotal
    wordErrorRate = sum(totalWER)/len(totalWER)
    print('Character error rate: %f%%. Address accuracy: %f%%. Word error rate: %f%%' %
          (charErrorRate*100.0, addressAccuracy*100.0, wordErrorRate*100.0))
    return charErrorRate, addressAccuracy, wordErrorRate


def load_different_image():
    imgs = []
    for i in range(1, Model.batchSize):
       imgs.append(preprocessor(cv2.imread("../data/check_image/a ({}).png".format(i), cv2.IMREAD_GRAYSCALE), Model.imgSize, enhance=False))
    return imgs


def generate_random_images():
    return np.random.random((Model.batchSize, Model.imgSize[0], Model.imgSize[1]))


def infer(model, fnImg):
    """ Recognize text in image provided by file path """
    img = preprocessor(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), imgSize=Model.imgSize)
    if img is None:
        print("Image not found")

    imgs = load_different_image()
    imgs = [img] + imgs
    batch = Batch(None, imgs)
    recognized = model.inferBatch(batch)  # recognize text

    print("Without Correction", recognized[0])
    print("With Correction", correct_sentence(recognized[0]))
    return recognized[0]



@eel.expose
def ocr(fname):

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'/home/chahal/Desktop/ocr/vision/ServiceAccountToken.json'
    client = vision.ImageAnnotatorClient()

    FOLDER_PATH = '/home/chahal/Desktop/test'
    IMAGE_FILE = fname
    FILE_PATH = os.path.join(FOLDER_PATH, IMAGE_FILE)

    with io.open(FILE_PATH, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)

    docText = response.full_text_annotation.text
    print(docText)

    eel.ocrreply(docText)

    file1=open("/home/chahal/Desktop/ocr/vision/IO_files/data.txt",'w+')
    file1.write(docText)
    file1.seek(0)
    if file1.read=="":
        exit(0)
    file1.close()
    index_main.main()

@eel.expose
def setnum(num):
    f1=open(r'/home/chahal/Desktop/AEVA/web/IO_files/setnum.txt','w+')
    f1.write(num)
    f1.close()

@eel.expose
def chat(text):
    reply=chatbot.chat(text)
    return reply

def main():
    """ Main function """
    # Opptional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", help="train the neural network", action="store_true")
    parser.add_argument(
        "--validate", help="validate the neural network", action="store_true")
    parser.add_argument(
        "--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # Train or validate on Cinnamon dataset
    if args.train or args.validate:
        # Load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize,
                            Model.imgSize, Model.maxTextLen, load_aug=True)

        # Execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=False)
            validate(model, loader)

    # Infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(),
                      decoderType, mustRestore=False)
        infer(model, FilePaths.fnInfer)


def infer_by_web(path, option):
    decoderType = DecoderType.BestPath
    print(open(FilePaths.fnAccuracy).read())
    model = Model(open(FilePaths.fnCharList).read(),
                  decoderType, mustRestore=False)
    recognized = infer(model, path)

    return recognized


     
eel.start('home.html')
    



    