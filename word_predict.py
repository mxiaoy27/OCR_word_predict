import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import keras
from PIL import Image
from keras.models import load_model


def get_label_dict():
    f = open('./b_label', 'rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict


lang_chars = get_label_dict()
words_label = []
for (char, value) in lang_chars.items():
    words_label.append(char)


class Prediction(object):
    def __init__(self, ModelFile, words_label, Width=25, Height=30):
        self.modelfile = ModelFile
        self.Width = Width
        self.Height = Height
        self.words_label = words_label

    def get_words_list(self, path_):
        list_name = []
        files = os.listdir(path_)
        files.sort()  # 对列表排序
        for word in files:
            word_path = os.path.join(path_, word)
            list_name.append(word_path)
        return list_name

    def Predict(self):
        keras.backend.clear_session()
        model = load_model(self.modelfile)

        image_set = []
        word_path = []
        word_path = self.get_words_list(path_='./image_temp/')
        # print(word_path)
        for image in word_path:
            new_img = Image.open(image).convert('L')
            new_img = new_img.resize((self.Width, self.Height), Image.ANTIALIAS)
            new_img.save(image)
            new_img = np.asarray(new_img)
            new_img = new_img.reshape([-1, 30, 25, 1])

            image_set.append(new_img)

        word_list = []
        for image_predict in image_set:
            prediction = model.predict(image_predict)
            Final_prediction = [result.argmax() for result in prediction][0]

            count = 0
            max_subscript = 0
            max_acc = 0.0
            for i in prediction[0]:
                if (max_acc < i):
                    max_acc = i
                    max_subscript = count
                count += 1
                # percentage1 = '%.2f%%' % (i * 100)
                # print (self.words_label[count-1],'概率:' ,percentage1)
            percentage = '%.2f%%' % (max_acc * 100)
            if (max_acc * 100) < 70.0:
                word_list.append(' ')
            else:
                word_list.append(self.words_label[max_subscript])
        sentence = "".join(word_list)
        print(sentence)
        # print (self.words_label[max_subscript],'概率:' ,percentage)

    def remove_(self):
        word_path = []
        word_path = self.get_words_list(path_='./image_temp/')
        for image in word_path:
            os.remove(image)

        word_path = self.get_words_list(path_='./row_cut/')
        for image in word_path:
            os.remove(image)

Pred = Prediction(ModelFile='words_recognize.h5',
                words_label=words_label,
                )
Pred.Predict()
#Pred.remove_()