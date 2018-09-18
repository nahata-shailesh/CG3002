import feature_extraction
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
    feature_extraction.stdev
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

def train_model(model):
    # train_segments = segmenting.load_segments('train_segments.txt')
    train_features = extract_features(train_segments[:, 0])
    train_labels = list(train_segments[:, 1])

    model.fit(train_features, train_labels)

    return model


def test_model(name, model):
    # test_segments = segmenting.load_segments('test_segments.txt')
    test_features = extract_features(test_segments[:, 0])
    test_labels = list(test_segments[:, 1])

    test_predictions = model.predict(test_features)

    print("%s: %f" % (name, accuracy_score(test_labels, test_predictions)))
    print(confusion_matrix(test_labels, test_predictions))
    print(classification_report(test_labels, test_predictions))


if __name__ == '__main__':
    
    for name, model in models:
    	train_model(model)
    	test_model(name, model)

