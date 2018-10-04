import feature_extraction
import segmentation
# import learning_curve
import numpy as np
# import matplotlib.pyplot as plt
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
models.append(('RF', RandomForestClassifier(n_estimators=128)))
models.append(('NN', MLPClassifier()))
# models.append(('ORC', OneVsRestClassifier(LinearSVC())))

def extract_features(segments):
    return feature_extraction.extract_features(segments, FEATURES_LIST)

def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train)
    return model

def test_model(name, model, X_validation, Y_validation):
    test_predictions = model.predict(X_validation)

    print("%s: %f" % (name, accuracy_score(Y_validation, test_predictions)))
    print(confusion_matrix(Y_validation, test_predictions))
    print(classification_report(Y_validation, test_predictions))

if __name__ == '__main__':
    segments = segmentation.load_segments('activity.csv', 200, 100)
    seg_features = []
    seg_labels = []

    for i in range(len(segments)):
        seg_features.append(segments[i][:, :3])
        seg_labels.append(segments[i][1, 3])
    
    features = extract_features(seg_features)
    labels = list(seg_labels)

    # print(np.array(features).shape)
    # print(np.array(labels).shape)

    validation_size = 0.20
    seed = 3
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(features, labels, test_size=validation_size, random_state=seed)
    # print(np.array(X_train).shape)
    # print(X_validation)
    # print(np.array(Y_train).shape)
    # print(Y_validation)


    for name, model in models:
        # train_model(model, X_train, Y_train)
        # test_model(name, model, X_validation, Y_validation)
        kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        msg = "%s: %f" % (name, cv_results.mean())
        print(msg)
        
        # plot_learning_curve(model, "Learning Curve", X_train, Y_train, ylim=(0.7, 1.01), cv=kfold, n_jobs=4)\
        # plt.show()


