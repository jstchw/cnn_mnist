import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
from keras.utils import plot_model

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# one-hot encode the labels
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)

# convolutional matrix reshape
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Input layer
inputs = Input(shape=x_train.shape[1:])

# Convolutional layer
conv = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(inputs)
conv = MaxPool2D(pool_size=(2, 2))(conv)
conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv)
conv = MaxPool2D(pool_size=(2, 2))(conv)
conv = Flatten()(conv)
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.5)(conv)

# Output layer
outputs = Dense(10, activation='softmax')(conv)

# Model
net = Model(inputs=inputs, outputs=outputs)

# Summary of the net
net.summary()

# Create loss function and optimizer
net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the network
history = net.fit(x_train, y_train, batch_size=256, epochs=20, validation_data=(x_test, y_test))

# Plot the loss and accuracy curves for training and validation

fig, ax = plt.subplots(1, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend1 = ax[0].legend(loc='best', shadow=True)
plt.show()

# Save the network
net.save('network_for_mnist.h5')
