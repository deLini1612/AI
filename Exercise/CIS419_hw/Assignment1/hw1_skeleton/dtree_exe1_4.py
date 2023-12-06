# Part 1.4

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def evaluatePerformance(num_trials=100, num_folds=10):
    # Define classifiers to evaluate
    classifiers = {
        'Decision Stump': tree.DecisionTreeClassifier(max_depth=1),
        '3-Level Decision Tree': tree.DecisionTreeClassifier(max_depth=3),
        'Depth-Unlimited Decision Tree': tree.DecisionTreeClassifier(),
        '5-Level Decision Tree': tree.DecisionTreeClassifier(max_depth=5),
        '10-Level Decision Tree': tree.DecisionTreeClassifier(max_depth=10)
    }

    # Initialize data structures for learning curves
    learning_curves = {name: [] for name in classifiers}
    train_percentages = np.arange(0.1, 1.1, 0.1)  # 10%, 20%, ..., 100%

    # Load and combine data
    data_train = np.loadtxt('data/SPECTF.train', delimiter=',')
    data_test = np.loadtxt('data/SPECTF.test', delimiter=',')
    data = np.vstack((data_train, data_test))

    for trial in range(num_trials):
        np.random.shuffle(data)
        kf = KFold(n_splits=num_folds)

        for fold, (train_index, test_index) in enumerate(kf.split(data)):
            train, test = data[train_index], data[test_index]
            Xtest, ytest = test[:, 1:], test[:, 0]

            for train_percentage in train_percentages:
                size = int(train_percentage * len(train))
                Xtrain_subset, ytrain_subset = train[:size, 1:], train[:size, 0]

                for name, clf in classifiers.items():
                    clf.fit(Xtrain_subset, ytrain_subset)
                    accuracy = accuracy_score(ytest, clf.predict(Xtest))
                    learning_curves[name].append((train_percentage, accuracy))

    # Compute mean and standard deviation for learning curves
    mean_std_curves = {name: {'mean': [], 'std': []} for name in classifiers}
    for name in classifiers:
        for percentage in train_percentages:
            accuracies = [acc for percent, acc in learning_curves[name] if percent == percentage]
            mean_std_curves[name]['mean'].append(np.mean(accuracies))
            mean_std_curves[name]['std'].append(np.std(accuracies))

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    for name in classifiers:
        means = mean_std_curves[name]['mean']
        stds = mean_std_curves[name]['std']
        plt.errorbar(train_percentages * 100, means, yerr=stds, label=name)

    plt.title("Learning Curves for Various Classifiers")
    plt.xlabel("Percentage of Training Data Used (%)")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    return mean_std_curves

if __name__ == "__main__":
    mean_std_curves = evaluatePerformance()
