from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
  
def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([-0.5,1])
    plt.ylim([-0.5,1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

class Classifier:
    def __init__(self, X, y, output_bias=None):
        
        METRICS = [
              keras.metrics.TruePositives(name='tp'),
              keras.metrics.FalsePositives(name='fp'),
              keras.metrics.TrueNegatives(name='tn'),
              keras.metrics.FalseNegatives(name='fn'), 
              keras.metrics.BinaryAccuracy(name='accuracy'),
              keras.metrics.Precision(name='precision'),
              keras.metrics.Recall(name='recall'),
              keras.metrics.AUC(name='auc'),
              keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]
        
        if output_bias is not None:
            output_bias=tf.keras.initializers.Constant(output_bias)
    
        self.train_X, self.test_X, self.val_X = X[0], X[1], X[2]
        self.train_y, self.test_y, self.val_y = y[0], y[1], y[2]
       

        self.model = Sequential()
        self.model.add(Input(shape=(29,)))
        self.model.add(Dense(22, activation='relu'))
        self.model.add(Dense(15, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(5, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS)
    
    def summary(self):
        print(self.model.summary())
        
    def train(self, batch_size, epochs, checkpoint_path, steps_per_epoch=None):
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq='epoch')
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_prc', 
            verbose=0,
            patience=10,
            mode='max',
            restore_best_weights=True)
        
        self.model.fit(self.train_X, self.train_y, batch_size=batch_size, epochs=epochs, validation_data=(self.val_X, self.val_y), callbacks=[early_stopping, cp_callback], verbose=0, steps_per_epoch=steps_per_epoch)
    
    def load(self, chk_dir):
        
        weights = tf.train.latest_checkpoint(chk_dir)
        self.model.load_weights(weights)
    
    def evaluate(self, batch_size):
    
        train_predictions_baseline = self.model.predict(self.train_X, batch_size=batch_size)
        test_predictions_baseline = self.model.predict(self.test_X, batch_size=batch_size)
        
        baseline_results = self.model.evaluate(self.test_X, self.test_y,
                                      batch_size=batch_size, verbose=0)
                                      
        for name, value in zip(self.model.metrics_names, baseline_results):
          print(name, ': ', value)
        print()
        
        y_pred = self.model.predict(self.test_X)
        y_pred = y_pred >= 0.5
        
        print()
        print("Classification Report:")
        print(classification_report(self.test_y, y_pred))
        
        cf_matrix = confusion_matrix(self.test_y, y_pred)
        
        print()
        print("Confusion Matrix:")
        print(cf_matrix)
        
        sns.heatmap(cf_matrix, annot=True)
        
        plt.show()
        plt.clf()
        plot_roc("Train Baseline", self.train_y, train_predictions_baseline, color='blue')
        plot_roc("Test Baseline", self.test_y, test_predictions_baseline, color='blue', linestyle='--')
        plt.show()
        plt.clf()
        
        plot_prc("Train Baseline", self.train_y, train_predictions_baseline, color='blue')
        plot_prc("Test Baseline", self.test_y, test_predictions_baseline, color='blue', linestyle='--')
        plt.show()
        plt.clf()