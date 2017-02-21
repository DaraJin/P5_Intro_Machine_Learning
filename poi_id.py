#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict
# A new feature created named 'from_poi_ratio'
# Set values to NaN if there's missing values
for i in my_dataset:
    if my_dataset[i]['from_this_person_to_poi'] != 'NaN' and my_dataset[i]['from_messages'] != 'NaN':
        my_dataset[i]['from_poi_ratio'] = float(my_dataset[i]['from_this_person_to_poi'])/ float(my_dataset[i]['from_messages'])
    else:
        my_dataset[i]['from_poi_ratio'] = 'NaN'

# Update features list

features_list = ['poi','from_poi_ratio', 'salary','deferred_income', 'expenses','total_stock_value',
                 'exercised_stock_options','long_term_incentive',  'director_fees', 'bonus']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# I tried 3 classifiers. 
# The scores were high because of overfitting.
if False:
  from sklearn.naive_bayes import GaussianNB
  clf = GaussianNB()
  clf.fit(features, labels)
  NB_s = clf.score(features, labels)

if False:
  from sklearn.neighbors import KNeighborsClassifier
  clf = KNeighborsClassifier()
  clf.fit(features, labels)
  KN_s = clf.score(features, labels)

if False:
  from sklearn import tree
  clf = tree.DecisionTreeClassifier()
  clf.fit(features, labels)
  DT_s = clf.score(features, labels)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

# Defining a function to tune parameters 
# and to sort out the best algorithm.
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

# SelectKBest, PCA used before tuning parameters 
# to achieve a better outcome. 
def scores(classifier, params):
    steps = [('SKB', SelectKBest(k = 5)), ('PCA', PCA(n_components=3)),('CLF', classifier)]
    pipe = Pipeline(steps)
    sss = StratifiedShuffleSplit(100, random_state = 42)
    search = GridSearchCV(pipe, params, cv = sss, scoring = "f1")
    search.fit(features, labels)
    clf = search.best_estimator_
    score = search.best_score_
    return clf, score

# Decision Tree Classifier
if False:
  from sklearn import tree
  classifier = tree.DecisionTreeClassifier()
  params = {'CLF__criterion': ('gini','entropy'),
            'CLF__min_samples_split': range(2,10), 
            'CLF__max_features': [2, 3],
            'CLF__max_depth': [5, 6, 7]}
  print "Decision Tree Classifier", scores(classifier, params)

# Random Forest
if False:
  from sklearn.ensemble import RandomForestClassifier
  classifer = RandomForestClassifier()
  params = {'CLF__criterion': ('gini','entropy'),
            'CLF__min_samples_split': range(2,10),
            'CLF__max_features': [2, 3], 
            'CLF__max_depth': [6, 7]}
  print "Random Forest", scores(classifier, params)

# KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
params = {'CLF__n_neighbors': range(1,5),
          'CLF__algorithm': ('ball_tree', 'kd_tree', 'brute')}
print "KNeighbors Classifier", scores(classifier, params)

clf = scores(classifier, params)[0]

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

# from tester import test_classifier
# print "Tester Classification report" 
# rst = test_classifier(clf, my_dataset, features_list)

  