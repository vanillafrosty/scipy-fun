from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()
data = breast_cancer_data.data
target = breast_cancer_data.target

subsets = train_test_split(data, target, train_size=0.8, random_state=120)
training_data, validation_data, training_labels, validation_labels = subsets

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(training_data, training_labels)


k_list = [i for i in range(1,101)]
accuracies = []

for k in k_list:
    classifier.n_neighbors = k
    score = classifier.score(validation_data, validation_labels)
    accuracies.append(score)


plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel("Validation Accuracy")
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
