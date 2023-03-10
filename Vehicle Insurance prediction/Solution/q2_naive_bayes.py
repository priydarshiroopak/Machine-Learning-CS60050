# -*- coding: utf-8 -*-
"""Naive_bayes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13k2-uCQJbLO19v3amKtV6Euw6eLmkfd6
"""

# Assignment 1
# Naive Bayes Classifier
# Group:
#   20CS30042 Roopak Priydarshi
#   20CS30046 Saras Umakant Pantulwar

import numpy as np
import pandas as pd

# Read data from *.csv file to pandas dataframe and analyse data
data = pd.read_csv('Dataset_C.csv')

# Display first five rows
display(data.head())
# Print the info
display(data.describe())
data.hist()
plt.show()

## GLOBAL variables for refining data ##
# Specify encoding for categorical variables:
g = {'Female': 0, 'Male': 1}
va = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
vd = {'No': 0, 'Yes': 1}
# Binning all values (continuous and discrete) to labels array
labels = [0,1,2]
  
def remove_outliers(data):
  """ Remove outliers for listed columns (having continuous data) """
  for col in ['Age',	'Region_Code',	'Annual_Premium',	'Policy_Sales_Channel']: 
    # Setting the min/max to outliers using standard deviation 
    factor = 3
    upper = data[col].mean () + data[col].std () * factor 
    lower = data[col].mean () - data[col].std () * factor 
    data = data[(data[col] < upper) & (data[col] > lower)]

def discretise(data):
  """ Allot the continuous values to discrete bins """
  for j in data.columns[:-1]:
      data[j] = pd.cut(data[j],bins=len(labels),labels=labels)

def preprocess(data):
  # Drop colums having ~Uniform distribution, they will not affect prediction
  data.drop(columns=['id', 'Vintage'], inplace=True)
  # Drop rows having null value in any column
  data.dropna(how='any',axis=0, inplace=True)
  # Encode categorical variables
  data.Gender = [g[val] for val in data.Gender.astype(str)]
  data.Vehicle_Age = [va[val] for val in data.Vehicle_Age.astype(str)]
  data.Vehicle_Damage = [vd[val] for val in data.Vehicle_Damage.astype(str)]
  # Drop Outlier rows
  remove_outliers(data)
  # convert to discrete bins:
  discretise(data)

preprocess(data)
# Display first five rows
display(data.head())
# Display the summary statistics
display(data.describe())

class NaiveBayes:
  """
    Bayes Theorem:
        Posterior Probability = (Likelihood * Class prior probability)/ Predictor Prior probability
                       P(c|x) = (P(x|c) * p(c))/ P(x)
  """
  def __init__(self, laplace = True, labels = [0,1,2]):
    """
      Attributes:
        features: List of features of the dataset
        likelihood: Likelihood of each feature per class
        class_prior: Prior probabilities of classes 
        pred_prior: Prior probabilities of features
    """
    self.features = np.array
    self.likelihood = {}
    self.class_prior = {}
    self.pred_prior = {}
    self.results = np.array
    self.X_train = np.array
    self.y_train = np.array
    self.train_size = int
    self.laplace = laplace
    self.alpha = 1
    self.labels = labels

  def train(self, X, y):
    # Initialise class variables
    self.X_train = X
    self.y_train = y
    self.features = self.X_train.columns
    self.feature_num = len(self.features)
    self.train_size = self.X_train.shape[0]
    self.results = np.unique(self.y_train)
    # Initialise model parameters
    for result in self.results:
      self.likelihood[result] = {}
      for feature in self.features:
        self.pred_prior[feature] = {}
        self.likelihood[result][feature] = {}
        for feat_val in self.labels:
          self.likelihood[result][feature].update({feat_val: 0})
          self.pred_prior[feature].update({feat_val: 0})

    # Set model parameters
    self._set_predictor_prior()
    self._set_likelihoods()
    self._set_class_prior()
    # print(self.likelihood)

  def _set_class_prior(self):
    """ Calculates Prior Class Probability - P(c) """
    for result in self.results:
      # count resulting values and update
      result_freq = sum(self.y_train == result)
      self.class_prior.update({result: result_freq / self.train_size})

  def _set_likelihoods(self):
    """ Calculates Likelihood - P(x|c) """
    for feature in self.features:
      # iterate over feature values
      for result in self.results:
        result_freq = sum(self.y_train == result)
        ft_likelihood = self.X_train[feature][self.y_train[self.y_train == result].index.values.tolist()].value_counts().to_dict()
        # update for each feature-value
        for ft_val in self.labels:
          if ft_likelihood.get(ft_val):
            freq = ft_likelihood[ft_val]
          else:
            freq = 0
          self.likelihood[result][feature].update({ft_val: (freq + self.laplace*self.alpha)/(result_freq + self.laplace*self.alpha*self.feature_num)})

  def _set_predictor_prior(self):
    """ Calculates Evidence - P(x) """
    for feature in self.features:
      # iterate over feature values
      ft_vals = self.X_train[feature].value_counts().to_dict()
      # update for each feature-value
      for ft_val in self.labels:
        if ft_vals.get(ft_val):
          freq = ft_vals[ft_val]
        else:
          freq = 0
      # for ft_val, freq in ft_vals.items():
        self.pred_prior[feature][ft_val] = (freq + self.laplace*self.alpha)/(self.train_size + self.laplace*self.alpha*self.feature_num)


  def predict(self, X):
    """ Predicts outputs based on calculation of Posterior probability P(c|x) """
    X = np.array(X)
    predictions = []
    # Iterate over every query
    for row in X:
      probs_prediction = {}
      # find probabilities for all valid results
      for result in self.results:
        prior = np.log(self.class_prior[result])
        likelihood = 0
        evidence = 0
        for feature, ft_val in zip(self.features, row):
          # calculate numerator
          likelihood += np.log(self.likelihood[result][feature][ft_val])
          # calculate denominator
          evidence += np.log(self.pred_prior[feature][ft_val])

        # posterior probability calculation:
        probs_prediction[result] = np.exp((likelihood + prior) - (evidence))
      # set the result for query as the one having maximum posterior probability
      predictions.append(max(zip(probs_prediction.values(), probs_prediction.keys()))[1])

    # return numpy array of predicted values
    return np.array(predictions)

def accuracy_score(y_true, y_pred):
	"""	score = (y_true - y_pred) / len(y_true) """
	return round(sum(y_pred == y_true)/len(y_true) * 100 ,3)

def partition(data):
	""" partioning data into features and target """
	X = data.drop([data.columns[-1]], axis = 1)
	y = data[data.columns[-1]]
	return X, y

def k_fold_validation(model, data, k = 10, shuffle = True):
  """ function to shuffle and partition data into k-folds, perform cross validation """
  if shuffle:
    shuffled = data.sample(frac=1).reset_index(drop=True)
  else:
    shuffled = data
  k = max(1, min(k, len(shuffled)))
  acc = []
  for i in range(k):
    train = shuffled.iloc[lambda x: x.index % k != i]
    test = shuffled.iloc[lambda x: x.index % k == i]
    X_train, y_train = partition(train)
    X_test, y_test = partition(test)
    model.train(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    acc.append(score)
    print('Fold %3d: [Training,Test] Split Distribution: [%6d, %5d], Accuracy: %.3f' % (i+1, len(y_train), len(y_test), score))
  print('\n%d-fold-Cross-Validation accuracy: %.3f +/- %.3f\n' %(k, np.mean(acc), np.std(acc)))


# ignore divide by 0 warnings for log likelihood calculations
np.errstate(divide='ignore')

# apply model without laplace transform
model2 = NaiveBayes(laplace = False, labels = labels)
k_fold_validation(model2, data)

# apply model with laplace transform
model1 = NaiveBayes(laplace = True, labels = labels)
k_fold_validation(model1, data)
