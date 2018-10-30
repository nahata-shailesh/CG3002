
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


# In[2]:


data_path = '/home/pi/Desktop/CG3002/training_data/feature_extracted_data/dataset7.csv'
label_path ='/home/pi/Desktop/CG3002/training_data/feature_extracted_data/label7.csv'


# In[3]:


data_df = pd.read_csv(data_path,header =None)
label_df = pd.read_csv(label_path,header=None)
label_df.head()


# In[5]:


data=np.asarray(data_df)
label=np.asarray(label_df).flatten('F') #change to 1D vector


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(data,label, test_size=0.2, random_state = 4)


# In[7]:


knn = KNeighborsClassifier(n_neighbors=3) 
gnb = GaussianNB()
rfc=RandomForestClassifier(random_state=4)
models=[]
models.append(knn)
models.append(gnb)
models.append(rfc)
kf = StratifiedKFold(n_splits=5, random_state = 4)


# In[8]:
rfc.fit(x_train, y_train)
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

y_pred=[]
for model in models:
    model.fit(x_train,y_train)
    y=model.predict(x_test)
    y_pred.append(y)  
    print(accuracy_score(y_test, y))


# In[9]:


# #K-fold validation , KNN -> GNB -> RFC
# for model in models:
#     scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='accuracy')
#     print(scores.mean())


# # In[10]:


# #confusion matrix  KNN -> GNB -> RFC
# #precsion : is intuitively the ability of the classifier 
# #not to label as positive a sample that is negative.
# #recall : to find all the positive samples.
# #fscore : balance betweeen precsion and score
# confusion_mat=[]
# for i in range(int(len(y_pred))):
#     print(confusion_matrix(y_test, y_pred[i]))
#     print(precision_recall_fscore_support(y_test, y_pred[i], average='micro'))
    
    


# # In[13]:


# target_names = ['rest', 'wiper', 'number7', 'chicken', 'sidestep', 'turnclap']
# for i in range(int(len(y_pred))):
#     print(classification_report(y_test, y_pred[i], target_names=target_names))

from sklearn.externals import joblib
joblib.dump(rfc, 'rfc_trained_3.joblib') 

