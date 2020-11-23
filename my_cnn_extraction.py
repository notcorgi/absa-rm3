from __future__ import print_function
import os
import numpy as np
import time

np.random.seed(1337)

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

path = r"C:\Users\lishubin\Desktop\github-repository\absa-rm3\final_data.csv"
csv_data = pd.read_csv(path, usecols=[0, 1], header=0, encoding='ansi')  # 读取csv文件，usecols为列索引号,header为列名所在行号
# print(csv_data['data'])
# print(csv_data['object'])
print('Processing text dataset')
sentences = list(csv_data['data'])
aspect = list(csv_data['object'])
print('Generated list of sentences..')
# print(type(list(sentences)))
print(len(sentences))  # 416

MAX_SEQ_LENGTH = 69
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300

print('Indexing word vectors.')

embeddings_index = {}
f = open('./glove.6B.300d.txt', encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
# print(embeddings_index)
print(list(embeddings_index.items())[:2])
print('Found %s word vectors.' % len(embeddings_index))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, lower=False)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print(word_index)
data = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
print(data)
print('Shape of data tensor:', data.shape)  # 2维数组，每一行代表一个句子，restaurants文件共有3044个句子

import nltk
from keras.preprocessing.text import text_to_word_sequence

# raw_output = corpus.findall('.//sentence')
# train_out = []
# delet = []
# print(data.shape)
# data = np.array(data)
# print(data.shape)
# i = 0
# for output in raw_output:
#     print('---')
#     s = text_to_word_sequence(output.find('text').text, lower=True)
#     print(s)
#     indices = np.zeros(MAX_SEQ_LENGTH)
#
#     aspectTerms = output.find('aspectTerms')
#     if (aspectTerms):
#         aspectTerm = aspectTerms.findall('aspectTerm')
#         k = 0
#         if (len(aspectTerm) > 0):
#             for aspect_term in aspectTerm:
#                 try:
#                     aspt = text_to_word_sequence(aspect_term.attrib['term'])
#                     print(aspt)  # 方面的列表：['orrechiete', 'with', 'sausage', 'and', 'chicken']
#                     if (len(aspt) < 2):  # term长的不要
#                         indices[s.index(aspt[0])] = 1
#                     else:
#                         k = 1
#                         break
#                 except:
#                     continue
#     else:
#         k = 1
#     if (k == 1):  # 剔除长的，空的term
#         delet.append(i)
#     train_out.append(indices)
#     i = i + 1
# print(train_out)
# print(train_out[0].shape)
# print(len(delet))  # 准备删除没有方面的数据
train_out = []
j = 0
for i in data:
    indices = np.zeros(MAX_SEQ_LENGTH)
    aspect_num = word_index[aspect[j]]
    aspect_index = list(i).index(aspect_num)
    print(aspect_index)
    indices[aspect_index] = 1
    train_out.append(indices)
    j = j + 1
    print(j)
print(train_out)

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = len(word_index)
embedding_matrix = np.zeros((nb_words + 1, 300))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)

# load pre-trained word embeddings into an Embedding layer
# Here, we have set trainable = False so as to keep the embeddings fixed.
embedding_layer = Embedding(nb_words + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTH,
                            trainable=False)
print('Embedding Layer set..')

from keras.models import Sequential

embedding_model = Sequential()
embedding_model.add(embedding_layer)
import keras.optimizers as opt

# optimizer = opt.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm=10, clipvalue=0)
# from keras import losses
# model.compile(loss=losses.mean_squared_error, optimizer='sgd')
embedding_model.compile(loss='categorical_crossentropy',
                        optimizer='RMSProp',
                        metrics=['acc']
                        )
embedding_output = embedding_model.predict(data)
print('Generated word Embeddings..')
print('Shape of Embedding_output', embedding_output.shape)

from keras.preprocessing.text import text_to_word_sequence
from nltk.tag.stanford import StanfordPOSTagger
from sklearn import preprocessing
from tqdm import tqdm

train_input = np.zeros(shape=(414, 69, 306))
le = preprocessing.LabelEncoder()
tags = ["CC", "NN", "JJ", "VB", "RB", "IN"]
le.fit(tags)
i = 0
# sentences = corpus.findall('.//sentence')
for sent in sentences:
    s = text_to_word_sequence(sent)  # 每一个句子的分词结果
    print(s)
    tags_for_sent = nltk.pos_tag(s)  # 词性标注
    #     print(tags_for_sent)
    sent_len = len(tags_for_sent)
    # ohe = [0] * 6
    #     print(ohe)

    for j in range(69):
        ohe = [0] * 6
        list = []
        if j < len(tags_for_sent) and tags_for_sent[j][1][:2] in tags:
            list.append(tags_for_sent[j][1][:2])
            #             print(list)
            ohe[le.transform(list)[0]] = 1
        train_input[i][j] = np.concatenate([embedding_output[i][j], ohe])  # 把词性追加在每个300维词向量的尾部，变成306维
    i = i + 1
print(train_input[2][3])
print(len(train_out))

# for i in sorted(delet, reverse=True):
#     j = j + 1
#     print(j)
#     train_input = np.delete(train_input, (i), axis=0)
#     train_out = np.delete(train_out, (i), axis=0)
train_out = np.array(train_out)
print('Concatenated Word-Embeddings and POS Tag Features...')
print(len(train_input), len(train_input[0]), len(train_input[0][0]))
print(len(train_out), len(train_out[0]))
print(type(train_out), type(train_out[0]))
print(train_out, train_input)

from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.preprocessing import sequence
# from keras.optimizers import RMSprop2,RMSprop
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_input, train_out, test_size=.3, random_state=0)
print('-----------------------------------------------')
print(X_train, y_train, X_test, y_test)

import keras.optimizers as opt

print('Training Model...')
model = Sequential()
# model.add(Convolution1D(100, 5, border_mode="same", input_shape=(69, 306)))
model.add(Convolution1D(100, 2, border_mode="same", input_shape=(69, 306)))
model.add(Activation("tanh"))
# model.add(MaxPooling1D(pool_length=5))
model.add(MaxPooling1D(pool_length=2))
model.add(Convolution1D(50, 3, border_mode="same"))
model.add(Activation("tanh"))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(500))  # 500
model.add(Activation("tanh"))
# softmax classifier
model.add(Dense(69, W_regularizer=l2(0.1)))
model.add(Activation("softmax"))

# model.load_weights('aspect_model_wepos.h5')
# import keras.optimizers as opt
#
# optimizer = opt.RMSprop()
model.compile(loss='msle',
# model.compile(loss='categorical_crossentropy',
              #               optimizer='RMSProp',
              optimizer='sgd',
              # optimizer='RMSProp',
              metrics=['acc']
              )
from matplotlib import pyplot

print('Model Trained.')
time_start = time.time()
# model.fit(train_input, train_out,
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    # cR,0.1,10,30,68.8%,14.6s,15.25s,'13.8s','15.5s'; cS,0.1,5,120,62.4%,64.04s,65.5s,'64.4s','63.5s'; mR,0.2,40,17,36.8%,5.04s,'5.16s','5.25';
                    batch_size=5,
                    nb_epoch=120,
                    )
time_end = time.time()

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['acc'])
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
my_y_ticks = np.arange(0, 13, 1)
pyplot.yticks(my_y_ticks)
pyplot.show()
# model.save_weights('aspect_wepos.h5')
model.save('m1.h5')
# y_pred = model.predict(train_input[:414])
y_pred = model.predict(X_test)
# print(train_input[:2739])
# print(y_pred)
# print(y_pred[0])
# print(y_pred[1])
# print(y_pred[2])
# print(y_pred[3])
# print('==================================')
# print(y_pred[len(y_pred)-1])
# print(y_pred[len(y_pred)-2])

#
# print(len(y_pred))  # 删除后余1301条数据
# print(len(y_pred[0]))

# processed_output = []
# for i in range(len(y_pred)):
#     processed_label =[]
#     for j in range(len(y_pred[i])):
#         if y_pred[i][j] > 0.1:
#             processed_label.append(1)
#         else:
#             processed_label.append(0)
#     processed_output.append(processed_label)
# print(processed_output)
# print(len(processed_output))
# print(len(processed_output[0]))
#
#
# def lookup(tokenizer, vec, returnIntNotWord=True):
#     twordkey = [(k, tokenizer.word_index[k]) for k in
#                 sorted(tokenizer.word_index, key=tokenizer.word_index.get, reverse=False)]
#     oneHotVec = []  # captures the index of the ords
#     engVec = []  # this one returns the indexs and the words. Make sure returnIntNotWord is false though
#     for eachRow, notUsed in enumerate(vec):
#         for index, item in enumerate(vec[0]):
#             if vec[eachRow][index] == 1:
#                 oneHotVec.append(index)
#     for index in oneHotVec:
#         engVec.append(twordkey[index])
#     if returnIntNotWord == True:
#         return oneHotVec
#     else:
#         return engVec
#
#
# # test_data = train_out[:414]
# test_data = y_test
# total_pos = 0.0
# true_pos = 0.0
# total_neg = 0.0
# true_neg = 0.0
# for i in range(len(test_data)):
#     for j in range(len(test_data[i])):
#         if test_data[i][j] == 1:
#             total_pos += 1
#             if processed_output[i][j] == 1:
#                 true_pos += 1
#             if processed_output[i][j] == 0:
#                 pass
#         #                 print(lookup(tokenizer,test_data[i],True))
#         if test_data[i][j] == 0:
#             total_neg += 1
#             if processed_output[i][j] == 0:
#                 true_neg += 1
#
# false_pos = total_neg - true_neg
# false_neg = total_pos - true_pos
# print(true_pos, true_neg, false_pos, false_neg)
#
# precision = true_pos/(true_pos+false_pos)
# recall = true_pos/total_pos
# f1_score = 2*precision*recall/(precision+recall)
# print ("precision - " +str(precision) + ", recall- " +str(recall)+ ", f1_score- " +str(f1_score))
# time_end=time.time()
# print('totally cost',time_end-time_start)

#

processed_output = []
for i in range(len(y_pred)):
    processed_label = []
    max = y_pred[i][0]
    index = 0
    for j in range(len(y_pred[i])):
        if y_pred[i][j] >= max:  ##################################################
            index = j
            max = y_pred[i][j]
        else:
            pass
    for n in range(len(y_pred[i])):
        if n == index:
            processed_label.append(1)
        else:
            processed_label.append(0)
    processed_output.append(processed_label)
print("processed_output")
print("y_test:")
for i in range(len(y_test)):
    print(y_test[i])
    print(processed_output[i])
    print(y_pred[i])
    print('----------------------------------')


def lookup(tokenizer, vec, returnIntNotWord=True):
    twordkey = [(k, tokenizer.word_index[k]) for k in
                sorted(tokenizer.word_index, key=tokenizer.word_index.get, reverse=False)]
    oneHotVec = []  # captures the index of the ords
    engVec = []  # this one returns the indexs and the words. Make sure returnIntNotWord is false though
    for eachRow, notUsed in enumerate(vec):
        for index, item in enumerate(vec[0]):
            if vec[eachRow][index] == 1:
                oneHotVec.append(index)
    for index in oneHotVec:
        engVec.append(twordkey[index])
    if returnIntNotWord == True:
        return oneHotVec
    else:
        return engVec


# test_data = train_out[:1301]
test_data = y_test
total_pos = 0.0
true_pos = 0.0
total_neg = 0.0
true_neg = 0.0
for i in range(len(test_data)):
    for j in range(len(test_data[i])):
        if test_data[i][j] == 1:
            total_pos += 1
            if processed_output[i][j] == 1:
                true_pos += 1
            if processed_output[i][j] == 0:
                pass
        #                 print(lookup(tokenizer,test_data[i],True))
        if test_data[i][j] == 0:
            total_neg += 1
            if processed_output[i][j] == 0:
                true_neg += 1

false_pos = total_neg - true_neg
false_neg = total_pos - true_pos
print(true_pos, true_neg, false_pos, false_neg)

precision = true_pos / (true_pos + false_pos)
recall = true_pos / total_pos
f1_score = 2 * precision * recall / (precision + recall)
acc = (true_pos + true_neg) / (total_pos + total_neg)
print("precision - " + str(precision) + ", recall- " + str(recall) + ", f1_score- " + str(f1_score))
print('totally cost', time_end - time_start)
print(history.history['loss'])
print(history.history['acc'])
print(history.history['val_acc'])
print(history.history['val_loss'])
# v = 0
# result_v = 0
# result_precision = 0
# result_recall = 0
# result_score = 0
# result_acc = 0
# result_processed_ouput = 0
# result_true_pos = 0
# result_true_neg = 0
# result_false_neg = 0
# result_false_pos = 0
# while v <= 1:
#     processed_output = []
#     for i in range(len(y_pred)):
#         processed_label = []
#         for j in range(len(y_pred[i])):
#             if y_pred[i][j] > v:  ##################################################
#                 processed_label.append(1)
#             else:
#                 processed_label.append(0)
#         processed_output.append(processed_label)
#     print(processed_output)
#     # print(len(processed_output))
#     # print(len(processed_output[0]))
#
#
#     def lookup(tokenizer, vec, returnIntNotWord=True):
#         twordkey = [(k, tokenizer.word_index[k]) for k in
#                     sorted(tokenizer.word_index, key=tokenizer.word_index.get, reverse=False)]
#         oneHotVec = []  # captures the index of the ords
#         engVec = []  # this one returns the indexs and the words. Make sure returnIntNotWord is false though
#         for eachRow, notUsed in enumerate(vec):
#             for index, item in enumerate(vec[0]):
#                 if vec[eachRow][index] == 1:
#                     oneHotVec.append(index)
#         for index in oneHotVec:
#             engVec.append(twordkey[index])
#         if returnIntNotWord == True:
#             return oneHotVec
#         else:
#             return engVec
#
#
#     # test_data = train_out[:1301]
#     test_data = y_test
#     # total_pos = 0.0
#     true_pos = 0.0
#     # total_neg = 0.0
#     true_neg = 0.0
#     false_neg = 0.0
#     false_pos = 0.0
#     for i in range(len(test_data)):
#         for j in range(len(test_data[i])):
#             if test_data[i][j] == 1:
#                 if processed_output[i][j] == 1:
#                     true_pos += 1
#                 if processed_output[i][j] == 0:
#                     false_neg += 1
#                 break
#             else:
#                 if processed_output[i][j] == 1:
#                     false_pos += 1
#                     break
#                 else:
#                     pass
#     true_neg = len(test_data) - true_pos - false_neg - false_pos
#     print(true_neg)
#     total_pos = true_pos + false_neg
#     total_neg = true_neg + false_pos
#     print(true_pos, true_neg, false_pos, false_neg)
#     if (true_pos + false_pos) == 0 or total_pos == 0:
#         break
#     precision = true_pos / (true_pos + false_pos)
#     recall = true_pos / total_pos
#     f1_score = 2 * precision * recall / (precision + recall)
#     acc = (true_pos + true_neg) / (total_pos + total_neg)
#     print("precision - " + str(precision) + ", recall- " + str(recall) + ", f1_score- " + str(f1_score))
#     if  acc > result_acc:
#         result_precision = precision
#         result_acc = acc
#         result_recall = recall
#         result_score = f1_score
#         result_v = v
#         result_processed_ouput = processed_output
#         result_false_neg = false_neg
#         result_false_pos = false_pos
#         result_true_neg = true_neg
#         result_true_pos = true_pos
#     v = v + 0.01
#     time_end = time.time()
#     print('totally cost', time_end - time_start)
