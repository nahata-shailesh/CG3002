import feature_extraction
import segmentation
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

FEATURES_LIST = [
    feature_extraction.energy,
    feature_extraction.entropy,
    feature_extraction.mean,
    feature_extraction.std_dev
]

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))
models.append(('NN', MLPClassifier()))
models.append(('ORC', OneVsRestClassifier(LinearSVC())))

def extract_features(segments):
    return feature_extraction.extract_features(segments, FEATURES_LIST)

def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train)
    return model

def test_model(name, model, X_validation, Y_validation):
    test_predictions = model.predict(X_validation)

    print("%s: %f" % (name, accuracy_score(Y_validation, test_predictions)))
    # print(confusion_matrix(Y_validation, test_predictions))
    # print(classification_report(Y_validation, test_predictions))

if __name__ == '__main__':
    segments = segmentation.load_segments('data.csv', 100, 50)
    seg_features = []
    seg_labels = []

    for i in range(len(segments)):
        seg_features.append(segments[i][:, :3])
        seg_labels.append(segments[i][1, 3])
    
    features = extract_features(seg_features)
    labels = list(seg_labels)

    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(features, labels, test_size=validation_size, random_state=seed)
    # print(X_train)
    # print(X_validation)
    # print(Y_train)
    # print(Y_validation)


    for name, model in models:
        train_model(model, X_train, Y_train)
        test_model(name, model, X_validation, Y_validation)
    #     kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #     cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    #     msg = "%s: %f" % (name, cv_results.mean())
    #     print(msg)

