from keras.models import load_model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import emnist

net1 = load_model('./models/model_1/network_1.h5')  # 50 epochs relu, 64 batch size
net2 = load_model('./models/model_2/network_2.h5')  # 100 epochs relu, 64 batch size
net3 = load_model('./models/model_3/network_3.h5')  # 45 epochs leaky relu (a=0.001), 64 batch size
net4 = load_model('./models/model_4/network_4.h5')  # 45 epochs relu, 32 batch size

nets = [net1, net2, net3, net4]

x_test, labels_test = emnist.extract_test_samples('byclass')

for i in range(len(nets)):
    outputs = nets[i].predict(x_test)
    labels_predicted = np.argmax(outputs, axis=1)
    missed = np.where(labels_predicted != labels_test)[0]

    # Print out the number of missed and correct predictions
    print(f'Missed {i+1}: %d' % len(missed))
    print(f'Correct {i+1}: %d' % (len(labels_test) - len(missed)))

    # Print out the percentage of correct predictions
    print(f'Accuracy {i+1}: %.2f%%' % (100 * (len(labels_test) - len(missed)) / len(labels_test)))
    # Print out the percentage of missed predictions
    print(f'Error {i+1}: %.2f%%' % (100 * len(missed) / len(labels_test)))

# plt.figure(figsize=(8, 2))
# for i in range(0, 8):
#     ax = plt.subplot(2, 8, i + 1)
#     plt.imshow(x_test[i, :].reshape(28, 28), cmap=plt.get_cmap('gray_r'))
#     plt.title(labels_test[i])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# for i in range(0, 8):
#     output = net.predict(x_test[i, :].reshape(1, 28, 28, 1))
#     output = output[0, 0:]
#     plt.subplot(2, 8, 8 + i + 1)
#     plt.bar(np.arange(10.), output)
#     plt.title(np.argmax(output))
#
# plt.show()
