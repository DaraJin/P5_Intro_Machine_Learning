
# Identify Fraud from Enron Email

## 1. Project Summary

#### Questions

Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

#### Answers

The goal of this project is to identify POI (Person of Interest) in the Enron fraud scandal by utilizing public Enron email dataset and financial dataset (E+F datasets). The Enron email dataset contains data from about 150 users mostly senior management of Enron. The corpus contains a total of about 0.5M messages. The financial dataset contains features of personal financial records like salary, bonus, stock etc. Together, the E+F datasets contain 146 data points, 21 features and 18 POI records. Given the limited POI records, by adopting advanced algorithms, machine learning appears to be the very technique to identify the different features between POIs and non-POIs and portray a persona of a POI.

Some outliers appeared when I tried visualising the dataset. One records named 'TOTAL' was removed and other outliers that showed extreme finance status were kept because it might indicate some patterns of POIs.

## 2. Feature Selection

#### Questions

What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

#### Answers

I ended up using 9 features of all including one that was newly added. When selecting features, firstly, I use my intuition to pick some important features and visualized the data to see if there were any patterns. Then, I created a new feature called 'from_poi_ratio' which indicates the proportion of emails sent from a POI among all received emails. This feature is useful in identifying the person with whom POIs were intended to contact with and be helpful to POIs. In that case, the person was very likely a POI himself. 

After creating the new feature to the dataset, I used SelectKBest to check all the variables and their scores. It turned out that the new feature I added was a top scorer with the score of 14.35 and 8 other feature scored above 1. I did no scaling. I did PCA before training the classifiers. PCA worked quite well in transforming the features. Scaling may worsen the outcome.

## 3. Algorithm

#### Questions

What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

#### Answers

I ended up using K Neighbors Classifier as the final algorithm. Before this, I also tried Random Forest and Decision Tree which was the base of Random Forest. As an ensembled algorithm, Random Forest did a slightly better job in getting random samples from original data and picking contributive features from all the samples. When introducing the K Neighbors Classifier, I found it not only got a better score in evaluation matrix but also ran faster than Random Forest and Decision Tree and took less effort in tuning parameters.

# 4.Tune Parameters

#### Questions

What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

#### Answers

It means to find certain values of the parameters that maximize the outcome of the algorithm. If the parameters were not well tuned, some may cause overfitting or run for quite a long time and others may give inaccurate outcomes. 

GridSearchCV was used to tune the parameters. Take Decision Tree as an example, the default parameters didn't fit the case very well. I decided to pick 3 most important parameters (min_samples_split, max_features and max_depth) and wrote a function based on GridSearchCV to test different parameter combinations by examining F1 scores. The function takes two inputs as parameters, one for classifier type, another for parameters options for the chosen classifier. The outcome is the best classifier with a set of best-fit parameters.

# 5. Validation

#### Questions

What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

#### Answers

Validation is a process to evaluate how well the algorithm works. A classic mistake is to use the same data for both training and testing features. It almost always causes overfitting. 

To avoid it, I hold out part of the available data as a test set that only used when testing. Taking out some data will definitely decrease the resources for training. To tackle the problem, I used stratified shuffle split cross-validation strategy as a pre-process of tuning parameters.

# 6. Evaluation Metrics

#### Questions

Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

#### Answers

###### KNeighbors Classifier

<img src="http://upload-images.jianshu.io/upload_images/2874338-abd349a4bd52a217.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"
style="width:300px;height:210px;float:left">

The matrix above shows the Evaluation outcome of KNeighbors Classifier. The upper blocks indicate the positive outcome after training the modified classifier. The left blocks show the data points that are True POIs in reality. Vice versa. The upper left block showes the classifier did a great job in predicting POIs. Given 15000 predictions, it correctly identified 732 POIs and failed to identify 821 POIs. The lower blocks show the negative ones. It means the classifier did a great job in predicting non-POIs. It correctly identified 12173 non-POIs and failed to identify 1268 non-POIs. 

The precision and recall rates are to describe how well the classifier does in predicting the POIs. Given the accuracy 0.86, under this algorithm, the precision is 0.47 and the recall rate is 0.37. These numbers are due to the limited records of POIs in original datasets.

###### Random Forest Classifier

<img src="http://upload-images.jianshu.io/upload_images/2874338-1f01db92e6c55dc2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"
style="width:300px;height:210px;float:left">

The same explanation can be applied to Random Forest Classifier. The accuracy rate is 0.85, the precision is 0.44 and the recall rate is 0.35. Comparatively speaking, KNeighbors Classifier did a better job in predicting the POIs overall.
