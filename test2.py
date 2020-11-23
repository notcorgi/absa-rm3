import tensorflow as tf
import keras.backend as K

list1 = []
list2 = []
list_update = []
iterations = K.variable(0, dtype='float32', name='iterations')
iterations1 = K.variable(1, dtype='float32', name='iterations')
iterations2=K.update_add(iterations1,1)
print(iterations1,iterations2)
list1.append(iterations)
list1.append(iterations1)
learning_rate = K.variable(0.01, name='learning_rate')
learning_rate2 = K.variable(0.02, name='learning_rate')
list2.append(learning_rate)
list2.append(learning_rate2)
for i, j in zip(list1, list2):
    list_update.append(K.update(i, j))
print(list1)
print(list2)
print(list_update)
