from keras.models import load_model
from keras.datasets import mnist
import numpy as np

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



