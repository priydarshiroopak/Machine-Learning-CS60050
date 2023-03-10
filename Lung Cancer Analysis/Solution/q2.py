# -*- coding: utf-8 -*-
"""q2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TluPg8-zog9SaWqplSBPmC7wf0UVpW4t
"""

# Assignment 2
# Support Vector Machine and Multilayer Perceptron
# Group:
#   20CS30042 Roopak Priydarshi
#   20CS30046 Saras Umakant Pantulwar

# library imports
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
# from random import sample     # if library function used for random sampling
from sklearn import clone
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# output redirection
sys.stdout=open('output_2.txt','w')

# Read data from file to pandas dataframe and analyse data
data = pd.read_csv('lung-cancer.data', header=None)

# Display first five rows
display(data.head())

def StandardScalarNorm(data):
  # apply Standard Scalar Normalisation to get values 
  # such that mean=0 and std. deviation=1 for every feature
  data = (data - data.mean()) / data.std()
  return data

def preprocess(data):
  # Mark rogue values as missing
  data = data.apply(pd.to_numeric, errors='coerce')
  # Fill missing values with median of column values
  data.fillna(data.median(numeric_only=True).round(1), inplace=True)
  Y = data[0]
  X = data.drop(0, axis=1)
  # Apply Standard Scalar Normalization
  X = StandardScalarNorm(X)
  # Encoding of categorical variables not required (all numerical values)
  return X, Y

def splitDataset(X, Y):
  size = len(data.index)
  # randomly split dataset in 20% - 80% ratio
  test_list = [x for x in range(size) if x%5==4]
  # test_list = sample(range(size), size//5)     # if library function used for random sampling
  # extract test data
  X_test = X[X.index.isin(test_list)]
  Y_test = Y[Y.index.isin(test_list)]
  # extract train data
  X_train = X[~X.index.isin(test_list)]
  Y_train = Y[~Y.index.isin(test_list)]
  return X_test, Y_test, X_train, Y_train

X, Y = preprocess(data)
X_test, Y_test, X_train, Y_train = splitDataset(X, Y)

def accuracy(Y_pred, Y_test):
  # create confusion matrix and find accuracy
  cm = confusion_matrix(Y_pred, Y_test)
  matrix_sum = cm.sum()
  diag_sum = cm.trace()
  return diag_sum / matrix_sum

# dictionary to store accuracies of SVMs
SVM_acc = {}

# initialise Linear SVM, fit and calculate accuracy
linear_SVM = SVC(kernel='linear')
linear_SVM.fit(X_train, Y_train)

Y_pred = linear_SVM.predict(X_test)
SVM_acc["Linear"] = accuracy(Y_pred, Y_test)

# initialise Quadratic SVM, fit and calculate accuracy
quad_SVM = SVC(kernel='poly', degree=2)
quad_SVM.fit(X_train, Y_train)

Y_pred = quad_SVM.predict(X_test)
SVM_acc["Quadratic"] = accuracy(Y_pred, Y_test)

# initialise Radial Basis fn. SVM, fit and calculate accuracy
radial_SVM = SVC(kernel='rbf')
radial_SVM.fit(X_train, Y_train)

Y_pred = radial_SVM.predict(X_test)
SVM_acc["Radial"] = accuracy(Y_pred, Y_test)

# show accuracies of different kernels in Support Vector Machine
print('Accuracies of SVM models over different kernels:\n')
for kernel, acc in SVM_acc.items():
  print("\tAccuracy of %s Kernel SVM : %.2f" %(kernel, acc))

plt.bar(SVM_acc.keys(), SVM_acc.values())
plt.xlabel('Kernels for SVMs')
plt.ylabel('Accuracy over test set')
plt.title('Accuracies of different SVMs')
plt.savefig("Accuracies over SVM kernels.png",dpi=100)
# plt.show()

# dictionaries to store models and accuracies for uni/bi-layered MLPs
MLP_models = {}
MLP_acc = {}

# initialise unilayer[16,] MLP, fit and calculate accuracy
MLP_models['unilayer'] = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, activation = 'relu', solver='sgd', random_state=0, learning_rate_init=0.001, batch_size=min(32, len(Y_train)))
MLP_models['unilayer'].fit(X_train, Y_train)

Y_pred = MLP_models['unilayer'].predict(X_test)
MLP_acc['unilayer'] = accuracy(Y_pred, Y_test)
print("\nAccuracy of MLPClassifier (single-layer- [16,]) : %.2f" %MLP_acc['unilayer'])

# initialise bilayer[256, 16] MLP, fit and calculate accuracy
MLP_models['bilayer'] = MLPClassifier(hidden_layer_sizes=(256, 16), max_iter=1000, activation = 'relu', solver='sgd', random_state=0, learning_rate_init=0.001, batch_size=min(32, len(Y_train)))
MLP_models['bilayer'].fit(X_train, Y_train)

Y_pred = MLP_models['bilayer'].predict(X_test)
MLP_acc['bilayer'] = accuracy(Y_pred, Y_test)
print("Accuracy of MLPClassifier (2-layer [256, 16]) : %.2f" %MLP_acc['bilayer'])

# select best model for further use
best_MLP = max(zip(MLP_acc.values(), MLP_acc.keys()))[1]
print("\nOut of the two models, %s MLP is selected due to higher accuracy(%.2f). \n" %(best_MLP, MLP_acc[best_MLP]))

# vary learning rate and choose best learning rate
lr = 1
MLP_lr_acc = {}
MLP_models[best_MLP].max_iter=4000
for i in range(5):
  lr = lr/10
  MLP_models[best_MLP].learning_rate_init = lr
  MLP_models[best_MLP].fit(X_train, Y_train)
  Y_pred = MLP_models[best_MLP].predict(X_test)
  MLP_lr_acc[str(lr)] = accuracy(Y_pred, Y_test)

# show accuracies of different learning rates in MLP
print('Accuracies of selected MLP model over learning rates:\n')
for lr, acc in MLP_lr_acc.items():
  print("\tAccuracy with learning rate %s : %.2f" %(lr, acc))
plt.bar(MLP_lr_acc.keys(), MLP_lr_acc.values())
plt.xlabel('Learning rates for MLP')
plt.ylabel('Accuracy over test set')
plt.title('Accuracies over different learning rates (MLP)')
plt.savefig("Accuracies over learning rates.png",dpi=100)
# plt.show()

# choose learning rate with best accuracy
best_lr = float(max(zip(MLP_lr_acc.values(), MLP_lr_acc.keys()))[1])
best_model = clone(MLP_models[best_MLP])
best_model.learning_rate_init = best_lr
best_model.max_iter = 500
best_model.fit(X_train, Y_train)
Y_pred = best_model.predict(X_test)
print("\nOut of the models, learning rate of %g is selected due to highest accuracy(%.2f). \n" %(best_lr, MLP_lr_acc[str(best_lr)]))

class SequentialForwardSelection():
  '''
  Instantiate with given model
  '''
  def __init__(self, model):
    self.model = clone(model)
        
  '''
  X_train - Training data Pandas dataframe
  X_test - Test data Pandas dataframe
  Y_train - Training label Pandas dataframe
  Y_test - Test label Pandas dataframe
  '''  
  def fit(self, X_train, X_test, Y_train, Y_test):
    total_features_count = X_train.shape[1]
    self.subsets_ = []
    self.scores_ = []
    self.indices_ = []

    # Find first feature that gives max model performance
    scores = []
    subsets = []
    for p in range(total_features_count):
      score = self._calc_acc(X_train.values, X_test.values, Y_train.values, Y_test.values, [p])
      subsets.append([p])
      scores.append(score)
    # Find the single feature having best score
    best_score_ind = np.argmax(scores)
    self.indices_ = list(subsets[best_score_ind])
    self.subsets_.append(self.indices_)
    self.scores_.append(scores[best_score_ind])
      
    # Add a feature one by one until accuracy doesn't decrease
    for dim in range(1, total_features_count):
      scores = []
      subsets = []
      current_feature = dim
      '''
      Add the remaining features one-by-one from the remaining feature set
      Calculate the score for every feature combinations
      '''
      for index in range(total_features_count):
        if index not in self.indices_:
          indices = list(self.indices_)
          indices.append(index)
          score = self._calc_acc(X_train.values, X_test.values, Y_train.values, Y_test.values,indices)
          subsets.append(indices)
          scores.append(score)

      # Get the index of best score
      best_score_ind = np.argmax(scores)
      if scores[best_score_ind] < self.scores_[-1]:
        break
      else:
        # Record best score
        self.scores_.append(scores[best_score_ind])
        # Get features which gave best score
        self.indices_ = list(subsets[best_score_ind])
        # Record the indices of features for best score
        self.subsets_.append(self.indices_)

    print('\nIterations of Sequential Forward Selector: \n')
    for features, score in zip(self.subsets_, self.scores_):
      print('\t%d best features give a score of %.2f' %(len(features), score))

    features = [X_train.columns.values[i] for i in self.indices_]
    print('\n Selected features: ', features)

    self.k_score_ = self.scores_[-1]
    return features
    
  '''
  Transform training, test data_set to
  data_set having selected features
  '''
  def transform(self, X):
    return X.values[:, self.indices_]
    
  '''
  Helper method to train model with specific set of features
  feature_ind = indices of features
  '''
  def _calc_acc(self, X_train, X_test, Y_train, Y_test, feature_ind):
    self.model.fit(X_train[:, feature_ind], Y_train)
    Y_pred = self.model.predict(X_test[:, feature_ind])
    acc = accuracy(Y_pred, Y_test)
    return acc

# Instantiate SequentialForwardSearch
sfs = SequentialForwardSelection(best_model)
# Fit data to determine features giving most optimal model performance
sfs.fit(X_train, X_test, Y_train, Y_test)
# Transform training dataset to dataset having selected features
X_train_sfs = sfs.transform(X_train)
# Transform the test dataset to dataset having selected features
X_test_sfs = sfs.transform(X_test)

class EnsembleLearner():
  ''' 
  Instantiate with given models
  '''
  def __init__(self, models):
    self.models = clone(models)
        
  '''
  X_train - Training data Pandas dataframe
  Y_train - Training label Pandas dataframe
  '''  
  def fit(self, X_train, Y_train):
    # train every model on the training data
    for model in self.models:
      model.fit(X_train, Y_train)

  '''
  X_test - Test data Pandas dataframe
  Y_pred - Returned predicted labels
  '''
  def predict(self, X_test):
    predictions = []
    for model in self.models:
      predictions.append(model.predict(X_test))
    # return the most voted label (modal value)
    return pd.DataFrame.from_records(predictions).mode().iloc[0]

models = [quad_SVM, radial_SVM, best_model]

# initialise ensemble learner with models
ensembleModel = EnsembleLearner(models)
# fit to train data
ensembleModel.fit(X_train, Y_train)
# predict and check accuracy for test data
Y_pred = ensembleModel.predict(X_test)

print('\nAccuracy of the max-voting based ensemble Learner of SVMs (Linear, Quadratic, Radial) and best MLP classifier is %.2f.' %accuracy(Y_pred, Y_test))