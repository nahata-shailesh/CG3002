import feature_extraction
import segmentation
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
    # feature_extraction.entropy,
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

def train_model(model):
    train_segments = segmentation.load_segments('data.csv', 200, 100)

    seg_features = []
    seg_labels = []

    for i in range(len(train_segments)):
    	seg_features.append(train_segments[i][:, :3])
    	seg_labels.append(train_segments[i][1, 3])

    train_features = extract_features(seg_features)
    train_labels = list(seg_labels)

    model.fit(train_features, train_labels)

    return model

def test_model(name, model):
	test_segments = segmentation.load_segments('data.csv', 200, 100)
	seg_features = []
	seg_labels = []
	for i in range(len(test_segments)):
		seg_features.append(test_segments[i][:, :3])
		seg_labels.append(test_segments[i][1, 3])

	test_features = extract_features(seg_features)
	test_labels = list(seg_labels)

	test_predictions = model.predict(test_features)

	print("%s: %f" % (name, accuracy_score(test_labels, test_predictions)))
    # print(confusion_matrix(test_labels, test_predictions))
    # print(classification_report(test_labels, test_predictions))

if __name__ == '__main__':
    
    for name, model in models:
    	train_model(model)
    	test_model(name, model)

