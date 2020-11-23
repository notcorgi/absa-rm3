import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from matplotlib import pyplot

def vectorize_sequences(sequences, dimensions=10000) :
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences) :
        results[i, sequence] = 1.
    return results

print('____Preparing data____')

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

print('____Building model____')

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))


model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.categorical_crossentropy,
              # loss=losses.binary_crossentropy,
              metrics=[metrics.categorical_accuracy])

              # loss=losses.mse,
              # metrics=['acc'])


print('____Training model____')

# history = model.fit(x_train[3000:],
#                     one_hot_train_labels[3000:],
#                     epochs=40,
#                     batch_size=20,#512
#                     validation_data=(x_train[:3000], one_hot_train_labels[:3000]))
history = model.fit(x_train,
                    one_hot_train_labels,
                    epochs=15,
                    batch_size=512,#512
                    validation_split=0.1)

print('____Evaluating model____')
pyplot.plot(history.history['val_loss'])
# pyplot.plot(history.history['val_categorical_accuracy'])
# pyplot.plot(history.history['categorical_accuracy'])
# pyplot.plot(history.history['acc'])
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
print(history.history['loss'])
print(history.history['val_loss'])
# print(history.history['acc'])
# print(history.history['val_acc'])
print(history.history['categorical_accuracy'])
print(history.history['val_categorical_accuracy'])



pyplot.show()
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

predictions = model.predict(x_test)
print(predictions)