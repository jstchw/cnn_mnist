import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import Model
from keras.layers import Conv2D, Dense, Flatten, Dropout, Input, BatchNormalization, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
import os

# Data augmenter
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest'
)

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# one-hot encode the labels
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)

# convolutional matrix reshape
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

datagen.fit(x_train)

# Input layer
inputs = Input(shape=x_train.shape[1:])

# Hidden layers 1
conv = Conv2D(filters=32, kernel_size=(3, 3), activation=LeakyReLU(alpha=0.01))(inputs)
conv = BatchNormalization()(conv)
conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv)
conv = BatchNormalization()(conv)
conv = Conv2D(filters=32, kernel_size=(5, 5), strides=2, padding='same', activation=LeakyReLU(alpha=0.01))(conv)
conv = BatchNormalization()(conv)
conv = Dropout(0.4)(conv)

conv = Conv2D(filters=64, kernel_size=(3, 3), activation=LeakyReLU(alpha=0.01))(conv)
conv = BatchNormalization()(conv)
conv = Conv2D(filters=64, kernel_size=(3, 3), activation=LeakyReLU(alpha=0.01))(conv)
conv = BatchNormalization()(conv)
conv = Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding='same', activation=LeakyReLU(alpha=0.01))(conv)
conv = BatchNormalization()(conv)
conv = Dropout(0.4)(conv)

conv = Conv2D(filters=128, kernel_size=(4, 4), activation='relu')(conv)
conv = BatchNormalization()(conv)
conv = Flatten()(conv)
conv = Dropout(0.4)(conv)

# Output layer with softmax
outputs = Dense(10, activation='softmax')(conv)

# Model
net = Model(inputs=inputs, outputs=outputs)

# Summary of the net
net.summary()

# Create loss function and optimizer
net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64

# Train the network
history = net.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size,
    epochs=45,
    validation_data=(x_test, y_test)
)

# Directory to search for folders
dir_path = './models/'

# Find highest numbered folder in directory
folder_names = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
if len(folder_names) > 0:
    latest_folder_name = max(folder_names)
    latest_folder_num = int(latest_folder_name.split('_')[-1])
else:
    latest_folder_num = 0

# Create new folder with next highest number
new_folder_num = latest_folder_num + 1
new_folder_name = 'model_{}'.format(new_folder_num)
os.makedirs(os.path.join(dir_path, new_folder_name))


# Save the network and all info
net.save(f'./{dir_path}/{new_folder_name}/network_{new_folder_num}.h5')
file = open(f'./{dir_path}/{new_folder_name}/history_{new_folder_num}.txt', 'w')
file.write(str(history.history))
file.write('\n')
file.write(str(history.params))
file.write('\n')
file.write(str(batch_size))
file.write('\n\n')
for layer in net.layers:
    file.write(str(layer.get_config()))
    file.write('\n')
file.close()

# Logging info

fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.savefig(f'./{dir_path}/{new_folder_name}/loss_{new_folder_num}.png')
plt.show()

