import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from autoencoder import Autoencoder
from classifier import Classifier
from utils import dataset_summary
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import tensorflow as tf

tf.random.set_seed(1)
np.random.seed(1)

def resample(X,y):
    #Resampling
    over = RandomOverSampler(sampling_strategy='minority')
    steps = [('o', over)]
    pipeline = Pipeline(steps=steps)
    train_X,train_y = pipeline.fit_resample(X,y)
    return train_X, train_y

def resample_under(X,y):
    #Resampling
    under = RandomUnderSampler(sampling_strategy='majority')
    steps = [('u', under)]
    pipeline = Pipeline(steps=steps)
    train_X,train_y = pipeline.fit_resample(X,y)
    return train_X, train_y

def split(X, y):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)
    
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.2)
    
    #Standard scaling
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    val_X = scaler.transform(val_X)
    
    train_X = np.clip(train_X, -5, 5)
    test_X = np.clip(test_X, -5, 5)
    val_X = np.clip(val_X, -5, 5)
    
    return [train_X, test_X, val_X], [train_y, test_y, val_y]

def read_dataset(csv_file_path, test_size=0.2, random_state=0):
    df = pd.read_csv(csv_file_path)
    df = df.drop(['Time'], axis=1)
    
    y = df['Class']
    X = df.drop(['Class'], axis=1)
    
    return X,y
    
if __name__ == '__main__':
    
    X,y = read_dataset('../creditcard.csv')
    # dataset_summary(X,y)
    X0, y0 = split(X,y)
    train_X, test_X, val_X = X0
    train_y, test_y, val_y = y0
    
    aenc = Autoencoder()
    noised_X = aenc.add_gausian_noise(train_X, mean=0, std=0.1)
    
    autoencoder_checkpoint_path = "autoencoder_checkpoints/cp-{epoch:04d}.ckpt"
    autoencoder_checkpoint_dir = os.path.dirname(autoencoder_checkpoint_path)
    
    aenc.train(noised_X, train_X, batch_size=2048, epochs=100, checkpoint_path=autoencoder_checkpoint_path)
    #aenc.load(autoencoder_checkpoint_dir)
    
    
    # test_X = aenc.add_gausian_noise(test_X, mean=0, std=0.1)
    train_X = aenc.predict(train_X)
    test_X = aenc.predict(test_X)
    val_X = aenc.predict(val_X)
    
    X1 = [train_X, test_X, val_X]
    
    classifier_checkpoint_path = "classifier_checkpoints/cp-{epoch:04d}.ckpt"
    classifier_checkpoint_dir = os.path.dirname(classifier_checkpoint_path)
    
    clf = Classifier(X1, y0)
    clf.train(batch_size=2048, epochs=100, checkpoint_path=classifier_checkpoint_path)
    #clf.load(classifier_checkpoint_dir)
    clf.evaluate(2048)
    
    ###############################################################################
    
    X,y = read_dataset('../creditcard.csv')
    # dataset_summary(X,y)
    X,y = resample(X,y)
    X0, y0 = split(X,y)
    train_X, test_X, val_X = X0
    train_y, test_y, val_y = y0
    
    aenc = Autoencoder()
    noised_X = aenc.add_gausian_noise(train_X, mean=0, std=0.1)
    
    autoencoder_checkpoint_path = "autoencoder_checkpoints/cp-{epoch:04d}.ckpt"
    autoencoder_checkpoint_dir = os.path.dirname(autoencoder_checkpoint_path)
    
    aenc.train(noised_X, train_X, batch_size=2048, epochs=100, checkpoint_path=autoencoder_checkpoint_path)
    #aenc.load(autoencoder_checkpoint_dir)
    
    
    # test_X = aenc.add_gausian_noise(test_X, mean=0, std=0.1)
    train_X = aenc.predict(train_X)
    test_X = aenc.predict(test_X)
    val_X = aenc.predict(val_X)
    
    X1 = [train_X, test_X, val_X]
    
    classifier_checkpoint_path = "classifier_checkpoints/cp-{epoch:04d}.ckpt"
    classifier_checkpoint_dir = os.path.dirname(classifier_checkpoint_path)
    
    clf = Classifier(X1, y0)
    clf.train(batch_size=2048, epochs=100, checkpoint_path=classifier_checkpoint_path)
    #clf.load(classifier_checkpoint_dir)
    clf.evaluate(2048)
    
    ###############################################################################
    
    X,y = read_dataset('../creditcard.csv')
    # dataset_summary(X,y)
    X,y = resample_under(X,y)
    X0, y0 = split(X,y)
    train_X, test_X, val_X = X0
    train_y, test_y, val_y = y0
    
    aenc = Autoencoder()
    noised_X = aenc.add_gausian_noise(train_X, mean=0, std=0.1)
    
    autoencoder_checkpoint_path = "autoencoder_checkpoints/cp-{epoch:04d}.ckpt"
    autoencoder_checkpoint_dir = os.path.dirname(autoencoder_checkpoint_path)
    
    aenc.train(noised_X, train_X, batch_size=2048, epochs=100, checkpoint_path=autoencoder_checkpoint_path)
    #aenc.load(autoencoder_checkpoint_dir)
    
    
    # test_X = aenc.add_gausian_noise(test_X, mean=0, std=0.1)
    train_X = aenc.predict(train_X)
    test_X = aenc.predict(test_X)
    val_X = aenc.predict(val_X)
    
    X1 = [train_X, test_X, val_X]
    
    classifier_checkpoint_path = "classifier_checkpoints/cp-{epoch:04d}.ckpt"
    classifier_checkpoint_dir = os.path.dirname(classifier_checkpoint_path)
    
    clf = Classifier(X1, y0)
    clf.train(batch_size=2048, epochs=100, checkpoint_path=classifier_checkpoint_path)
    #clf.load(classifier_checkpoint_dir)
    clf.evaluate(2048)
    

##Tensorboard run command
#tensorboard --logdir logs/fit
    