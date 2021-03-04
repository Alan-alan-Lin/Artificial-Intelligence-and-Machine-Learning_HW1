import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation

batch_size = 128
num_classes = 10
epochs = 10

#切割成訓練集和測試集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#圖片預處理
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255
print('x_train shape', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#convert class vectors to binary matrics
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model
model = Sequential()
model.add(Conv2D(64, (3, 3), border_mode='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), border_mode='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

#Training
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
