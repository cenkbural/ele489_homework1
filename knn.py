import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



columns = [
    'class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

df = pd.read_csv("wine.data", header=None, names=columns)

print(df.head())

print(df.isnull().sum())

features = df.columns[1:] 
classes = df['class'].unique()


plt.figure(figsize=(20, 30))

for i, feature in enumerate(features):
    plt.subplot(5, 3, i + 1)
    for cls in classes:
        subset = df[df['class'] == cls]
        sns.kdeplot(subset[feature], label=f"Class {cls}", fill=True, alpha=0.3)
    plt.title(f"{feature} by Class")
    plt.legend()

plt.tight_layout()
plt.show()



sns.pairplot(df, hue='class', vars=[
    'alcohol', 'flavanoids', 'color_intensity', 'od280/od315_of_diluted_wines', 'proline'
])
plt.show()


plt.figure(figsize=(20, 30))

for i, feature in enumerate(features):
    plt.subplot(5, 3, i + 1)
    sns.boxplot(x='class', y=feature, data=df)
    plt.title(f"{feature} by Class")

plt.tight_layout()
plt.show()

X = df.drop('class', axis=1)
y = df['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])


def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def knn_predict(X_train, y_train, X_test, k=3, distance_metric='euclidean'):
    predictions = []
    
    for x in X_test:
        if distance_metric == 'euclidean':
            distances = [euclidean(x, x_train) for x_train in X_train]
        elif distance_metric == 'manhattan':
            distances = [manhattan(x, x_train) for x_train in X_train]
        else:
            raise ValueError("Unknown distance metric")

        k_indices = np.argsort(distances)[:k]

        k_nearest_labels = [y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    
    return predictions


k_values = list(range(1, 31))  # [1, 3, 5, ..., 21]

accuracies_euclidean = []
accuracies_manhattan = []

for k in k_values:
    # using Euclidean distance
    y_pred_euc = knn_predict(X_train, y_train.tolist(), X_test, k=k, distance_metric='euclidean')
    acc_euc = accuracy_score(y_test, y_pred_euc)
    accuracies_euclidean.append(acc_euc)
    
    # using Manhattan distance
    y_pred_man = knn_predict(X_train, y_train.tolist(), X_test, k=k, distance_metric='manhattan')
    acc_man = accuracy_score(y_test, y_pred_man)
    accuracies_manhattan.append(acc_man)

plt.plot(k_values, accuracies_euclidean, marker='o', label='Euclidean')
plt.plot(k_values, accuracies_manhattan, marker='s', label='Manhattan')
plt.title('Accuracy vs K')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

def plot_confusion_matrices(k_values, X_train, y_train, X_test, y_test):
    for distance in ['euclidean', 'manhattan']:
        for k in k_values:
            y_pred = knn_predict(X_train, y_train.tolist(), X_test, k=k, distance_metric=distance)

            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix (k={k}, {distance.capitalize()})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.show()

            print(f"Classification Report (k={k}, {distance.capitalize()}):")
            print(classification_report(y_test, y_pred))

plot_confusion_matrices(k_values, X_train, y_train, X_test, y_test)


# K values to test
k_values = list(range(1, 31))  # 1 to 30 inclusive

# Accuracy lists
accuracies_euclidean = []
accuracies_manhattan = []

# Loop over each K
for k in k_values:
    # Euclidean distance (default = 'minkowski', p=2)
    knn_euc = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    knn_euc.fit(X_train, y_train)
    y_pred_euc = knn_euc.predict(X_test)
    acc_euc = accuracy_score(y_test, y_pred_euc)
    accuracies_euclidean.append(acc_euc)

    # Manhattan distance (p=1)
    knn_man = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=1)
    knn_man.fit(X_train, y_train)
    y_pred_man = knn_man.predict(X_test)
    acc_man = accuracy_score(y_test, y_pred_man)
    accuracies_manhattan.append(acc_man)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies_euclidean, marker='o', label='Euclidean (p=2)')
plt.plot(k_values, accuracies_manhattan, marker='s', label='Manhattan (p=1)')
plt.title('Accuracy vs K (Sklearn KNN)')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

k_values = list(range(1, 31))
comparison_results = []

for k in k_values:
    # --- Sklearn KNN ---
    clf = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    clf.fit(X_train, y_train)
    y_pred_sklearn = clf.predict(X_test)

    # --- Custom KNN ---
    y_pred_custom = knn_predict(X_train, y_train.tolist(), X_test, k=k, distance_metric='euclidean')

    # --- Accuracy Scores ---
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    acc_custom = accuracy_score(y_test, y_pred_custom)

    # --- Prediction Match Rate ---
    match_count = np.sum(np.array(y_pred_custom) == np.array(y_pred_sklearn))
    total = len(y_test)
    match_rate = match_count / total

    comparison_results.append({
        'k': k,
        'acc_sklearn': acc_sklearn,
        'acc_custom': acc_custom,
        'match_rate': match_rate,
        'different_predictions': total - match_count
    })

comparison_df = pd.DataFrame(comparison_results)
print(comparison_df)

plt.figure(figsize=(10, 6))
plt.plot(comparison_df['k'], comparison_df['acc_sklearn'], label='Sklearn Accuracy', marker='o')
plt.plot(comparison_df['k'], comparison_df['acc_custom'], label='Custom Accuracy', marker='s')
plt.plot(comparison_df['k'], comparison_df['match_rate'], label='Prediction Match Rate', linestyle='--', color='gray')
plt.xlabel('K')
plt.ylabel('Score')
plt.title('Custom vs Sklearn KNN Comparison')
plt.legend()
plt.grid(True)
plt.show()