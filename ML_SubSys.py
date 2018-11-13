
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier


# In[2]:


data_path = '/Users/tingtingx/Documents/CEG4/SEM1/CG3002/ML_Code/datasetSetF.csv'
label_path ='/Users/tingtingx/Documents/CEG4/SEM1/CG3002/ML_Code/labelSetF.csv'


# In[3]:


data_df = pd.read_csv(data_path,header =None)
label_df = pd.read_csv(label_path,header=None)
data_df.isnull().sum().sum()
scaler = StandardScaler()
scaler.fit(data_df)
data_df=scaler.transform(data_df)


# In[4]:


data=np.asarray(data_df)
label=np.asarray(label_df).flatten('F') #change to 1D vector


# In[5]:


x_train, x_test, y_train, y_test = train_test_split(data,label, test_size=0.2, random_state = 4)


# In[6]:


mlp =MLPClassifier(random_state=4)
gnb = GaussianNB()

rfc=RandomForestClassifier(random_state=4)
rfc2=RandomForestClassifier(random_state=4)
ovr = OneVsRestClassifier(mlp)
models =[]
models.append(ovr)
models.append(mlp)
models.append(rfc)
kf = StratifiedKFold(n_splits=5, random_state = 4)


# In[7]:


y_pred=[]
for model in models:
    model.fit(x_train,y_train)
    y=model.predict(x_test)
    y_pred.append(y)  
    print(accuracy_score(y_test, y))


# In[8]:


#K-fold validation , KNN -> GNB -> RFC
for model in models:
    scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='accuracy')
    print(scores.mean())


# In[9]:


#confusion matrix  KNN -> GNB -> RFC
#precsion : is intuitively the ability of the classifier 
#not to label as positive a sample that is negative.
#recall : to find all the positive samples.
#fscore : balance betweeen precsion and score
confusion_mat=[]
for i in range(int(len(y_pred))):
    print(confusion_matrix(y_test, y_pred[i]))
    print(precision_recall_fscore_support(y_test, y_pred[i], average='micro'))
    
    


# In[10]:


target_names = ['rest', 'wiper', 'number7', 'chicken', 'sidestep', 'turnclap','number6','salute','mermaid','swing','cowboy','logout']
for i in range(int(len(y_pred))):
    print(classification_report(y_test, y_pred[i], target_names=target_names))


# In[11]:


importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[12]:


# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curveA# from sklearn.model_selection import ShuffleSplit


# In[13]:


# #taken from scikit learn
# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     """
#     Generate a simple plot of the test and training learning curve.

#     Parameters
#     ----------
#     estimator : object type that implements the "fit" and "predict" methods
#         An object of that type which is cloned for each validation.

#     title : string
#         Title for the chart.

#     X : array-like, shape (n_samples, n_features)
#         Training vector, where n_samples is the number of samples and
#         n_features is the number of features.

#     y : array-like, shape (n_samples) or (n_samples, n_features), optional
#         Target relative to X for classification or regression;
#         None for unsupervised learning.

#     ylim : tuple, shape (ymin, ymax), optional
#         Defines minimum and maximum yvalues plotted.

#     cv : int, cross-validation generator or an iterable, optional
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:
#           - None, to use the default 3-fold cross-validation,
#           - integer, to specify the number of folds.
#           - An object to be used as a cross-validation generator.
#           - An iterable yielding train/test splits.

#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is not a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validators that can be used here.

#     n_jobs : integer, optional
#         Number of jobs to run in parallel (default 1).
#     """
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     return plt


# In[14]:


# title = "Learning Curves"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# plot_learning_curve(rfc, title, x_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

# plt.show()


# In[15]:


from sklearn.externals import joblib
joblib.dump(rfc, 'rfc_trained.joblib') 
joblib.dump(mlp, 'mlp_trained.joblib')
joblib.dump(ovr, 'ovr_trained.joblib')

