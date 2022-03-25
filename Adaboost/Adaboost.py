import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import sklearn

np.random.seed(1)

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils import dataset_summary

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
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

class AdaBoost:
    def __init__(self):  
        self.model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=200, random_state=0)
        
    def train(self, X, y, checkpoint_dir):
        self.model.fit(X, y)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        
        with open(checkpoint_dir + "checkpoint.chk", 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, checkpoint_dir):
        if not os.path.isfile(checkpoint_dir + "checkpoint.chk"):
            print("No model found!")
            return

        with open(checkpoint_dir + "checkpoint.chk", 'rb') as f:
            self.model = pickle.load(f)

    def evaluate(self, train_X, train_y, test_X, test_y):
        
        y_pred = self.model.predict(test_X)
        y_pred = y_pred >= 0.5
        
        print()
        print("Classification Report:")
        print(classification_report(test_y, y_pred))
        
        cf_matrix = confusion_matrix(test_y, y_pred)
        
        print()
        print("Confusion Matrix:")
        print(cf_matrix)
        
        sns.heatmap(cf_matrix, annot=True)
        
        plt.show()
        plt.clf()
        
        train_predictions_baseline = self.model.predict(train_X)
        test_predictions_baseline = self.model.predict(test_X)
        
        plot_roc("Train Baseline", train_y, train_predictions_baseline, color='blue')
        plot_roc("Test Baseline", test_y, test_predictions_baseline, color='blue', linestyle='--')
        plt.show()
        plt.clf()
        
        plot_prc("Train Baseline", train_y, train_predictions_baseline, color='blue')
        plot_prc("Test Baseline", test_y, test_predictions_baseline, color='blue', linestyle='--')
        plt.show()
        plt.clf()
        
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
    X,y= split(X,y)
    #dataset_summary(X,y)
    train_X, test_X, val_X = X
    train_y, test_y, val_y = y
    
    checkpoint_dir = "checkpoints/"
    
    model = AdaBoost()
    model.train(train_X, train_y, checkpoint_dir=checkpoint_dir)
    # model.load(checkpoint_dir)
    model.evaluate(train_X, train_y, test_X, test_y)
    
    ####################################################################
    
    X,y = read_dataset('../creditcard.csv')
    X,y = resample(X,y)
    X,y = split(X,y)
    #dataset_summary(X,y)
    train_X, test_X, val_X = X
    train_y, test_y, val_y = y
    
    checkpoint_dir = "checkpoints/"
    
    model = AdaBoost()
    model.train(train_X, train_y, checkpoint_dir=checkpoint_dir)
    # model.load(checkpoint_dir)
    model.evaluate(train_X, train_y, test_X, test_y)
    
    ####################################################################
    
    X,y = read_dataset('../creditcard.csv')
    X,y = resample_under(X,y)
    X,y = split(X,y)
    #dataset_summary(X,y)
    train_X, test_X, val_X = X
    train_y, test_y, val_y = y
    
    checkpoint_dir = "checkpoints/"
    
    model = AdaBoost()
    model.train(train_X, train_y, checkpoint_dir=checkpoint_dir)
    # model.load(checkpoint_dir)
    model.evaluate(train_X, train_y, test_X, test_y)