from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
import os

bottleneck_size = 2

input_img = Input(shape=(784,))

encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(bottleneck_size, activation='linear')(encoded)
encoder = Model(input_img, encoded)


encoded_input = Input(shape=(bottleneck_size,))
decoded = Dense(64, activation='relu')(encoded_input)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
decoder = Model(encoded_input, decoded)

full = decoder(encoder(input_img))
ae = Model(input_img, full)
ae.compile(optimizer='adam', loss='mean_squared_error')


######
from keras.datasets import mnist
import numpy as np
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32')  / 255.
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
np.save("x_train", x_train)
np.save("y_train", y_train)
np.save("x_test", x_test)
np.save("y_test", y_test)
'''

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test =  np.load("x_test.npy")
y_test =  np.load("y_test.npy")


###########
if "model.h5" in os.listdir():
    ae = load_model('model.h5')
    encoder = load_model('encoder.h5')
    decoder = load_model('decoder.h5')
else:
    ae.fit(x_train, x_train, 
        epochs = 10,
        batch_size=256,
        validation_data=(x_test, x_test))
    ae.save('model.h5')
    encoder.save('encoder.h5')
    decoder.save('decoder.h5')


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

'''
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],
	c=y_test, s=8, cmap='tab10')
plt.show() 


