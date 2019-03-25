import pylab as plt
import numpy as np

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.decomposition import PCA


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255


k= 2
n_sample = 60000

#PCA
'''
mu = x_train.mean(axis=0)
U, s, V = np.linalg.svd(x_train - mu, full_matrices=False)
Zpca = np.dot(x_train - mu, V.transpose())

Rpca = np.dot(Zpca[:, :2], V[:2, :]) + mu    # reconstruction
err = np.sum((x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
print('PCA reconstruction error with 2 PCs: ' + str(round(err, 3)))
'''

pca = PCA(n_components=k)
pca.fit(x_train)
Zpca = pca.transform(x_train)



#AutoEncoder
ae = Sequential()
ae.add(Dense(512,  activation='relu', input_shape=(784,)))
ae.add(Dense(128,  activation='relu'))
ae.add(Dense(2,    activation='linear', name="bottleneck"))
ae.add(Dense(128,  activation='relu'))
ae.add(Dense(512,  activation='relu'))
ae.add(Dense(784,  activation='sigmoid'))
ae.compile(loss='mean_squared_error', optimizer=Adam())


def explained_variance(projections):
    var = np.array([(projections[:,i]**2).sum()/(n_sample-1) for i in range(k)]).round(2)
    return var


def fit():
    history = ae.fit(x_train, x_train, batch_size=128, epochs=1,
                 verbose=1, validation_data=(x_test, x_test))

def encode_fig(i):
    encoder = Model(ae.input, ae.get_layer('bottleneck').output)
    Zenc = encoder.predict(x_train)
    print("At epoch ", i, ", explained variance: ", explained_variance(Zenc)) 
    plt.title('Autoencoder')
    plt.scatter(Zenc[:5000, 0], Zenc[:5000, 1],
            c=y_train[:5000], s=8, cmap='tab10')
    plt.savefig("./gif/"+str(i)+".png")
    plt.clf()
   

for i in range(0, 60000, 10000):
    ae.fit(x_train[i:i+10000], x_train[i:i+10000], batch_size=128, epochs=1,
            verbose=1, validation_data=(x_test, x_test))
    encode_fig(-70000+i)

for i in range(20):
    fit()
    encode_fig(i) 

encoder = Model(ae.input, ae.get_layer('bottleneck').output)
Zenc = encoder.predict(x_train)




print("PCA explained variance:", pca.explained_variance_.round(2))

#percent_explained = [explained_var[i]/explained_var.sum() for i in range(k)]
#print("PCA self calculated:", percent_explained)


print("Pca explained variance:", explained_variance(Zpca))
print("ae explained variance:", explained_variance(Zenc))


# PLOT
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.title('PCA')
plt.scatter(Zpca[:5000, 0], Zpca[:5000, 1],
            c=y_train[:5000], s=8, cmap='tab10')

plt.subplot(122)
plt.title('Autoencoder')
plt.scatter(Zenc[:5000, 0], Zenc[:5000, 1],
            c=y_train[:5000], s=8, cmap='tab10')

plt.tight_layout()
plt.show()
