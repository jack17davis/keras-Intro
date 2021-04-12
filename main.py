import keras
%pylab inline
#import matplotlib
#import numpy as np

#
# 1.1. Load Data
#
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

#
# 1.2. Data Manipulation
# Convert pixel values to 0...1 range
#
X_train = X_train.astype(np.float32)
X_train /= 255
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
y_train = keras.utils.to_categorical(y_train, 10)

X_test = X_test.astype(np.float32)
X_test /= 255
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
y_test = keras.utils.to_categorical(y_test, 10)

#
# testing to see how things came out
#
imshow(X_train[1,:,:,0], cmap='gray')
y_train[1]

#
# 2. Setup Network
#
first_image = X_train[0]
NUMBER_OF_CLASSES = 10
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                                 activation='relu',
                                                              input_shape=first_image.shape))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax'))

# Compile the Network
model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                                    metrics=['accuracy'])

#
# 3. Training
#
model.fit(X_train, y_train)

#
# Testing
#
prediction = model.predict(X_test[1:2])
np.argmax(prediction)
