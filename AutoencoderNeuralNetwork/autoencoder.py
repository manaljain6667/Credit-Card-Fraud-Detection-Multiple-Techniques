from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import numpy as np

class Autoencoder():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Input(shape=(29,)))
        self.model.add(Dense(22, activation='relu'))
        self.model.add(Dense(15, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(15, activation='relu'))
        self.model.add(Dense(22, activation='relu'))
        self.model.add(Dense(29, activation='relu'))
        
        def lf(pred, orig):
            reconstruction_error = tf.reduce_mean(0.5 * tf.square(tf.subtract(pred, orig)))
            return reconstruction_error
        
        self.loss_function = lf
        self.model.compile(optimizer='adam', loss = tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
        
    def summary(self):
        print(self.model.summary())
    
    def train(self, X, y, batch_size, epochs, checkpoint_path):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq='epoch')
        
        self.model.fit(X,y,batch_size=batch_size, epochs=epochs, callbacks=[cp_callback])
        
    def load(self, chk_dir):   
        weights = tf.train.latest_checkpoint(chk_dir)
        self.model.load_weights(weights)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def add_gausian_noise(self, X, mean, std):
        noise = np.random.normal(mean,std,X.shape)
        new_X = X + noise
        return new_X        