from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

net = load_model('network_for_mnist.h5')

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

outputs = net.predict(x_test)

labels_predicted = np.argmax(outputs, axis=1)
missed = np.where(labels_predicted != labels_test)[0]

# Print out the number of missed and correct predictions
print('Missed: %d' % len(missed))
print('Correct: %d' % (len(labels_test) - len(missed)))

# Print out the percentage of correct predictions
print('Accuracy: %.2f%%' % (100 * (len(labels_test) - len(missed)) / len(labels_test)))
# Print out the percentage of missed predictions
print('Error: %.2f%%' % (100 * len(missed) / len(labels_test)))


plt.figure(figsize=(8, 2))
for i in range(0, 8):
    ax = plt.subplot(2, 8, i + 1)
    plt.imshow(x_test[i, :].reshape(28, 28), cmap=plt.get_cmap('gray_r'))
    plt.title(labels_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
for i in range(0, 8):
    output = net.predict(x_test[i, :].reshape(1, 28, 28, 1))
    output = output[0, 0:]
    plt.subplot(2, 8, 8 + i + 1)
    plt.bar(np.arange(10.), output)
    plt.title(np.argmax(output))

plt.show()
