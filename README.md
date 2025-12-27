# Detecting-Fake-Bills
This project leverages descriptive data mining techniques on a dataset pertaining to counterfeit bills, utilizing the k-nearest neighbors algorithm to discern whether a bill is legitimate or counterfeit. The k-nearest neighbor algorithm was selected due to its flexibility, simplicity, and applicability to descriptive data mining tasks. 

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
url = 'https://www.kaggle.com/datasets/alexandrepetit881234/fake-bills/data'
df = pd.read_csv(url)  # Replace with the actual local path or URL if available

# Clean data
df = df.dropna()

# Encode categorical columns if necessary (example: 'is_genuine' column)
label_encoder = LabelEncoder()
df['is_genuine'] = label_encoder.fit_transform(df['is_genuine'])

# Define features and target variable
X = df.drop('is_genuine', axis=1)
y = df['is_genuine']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize KNN classifier with k=3, as suggested
knn = KNeighborsClassifier(n_neighbors=3)

# Train the KNN classifier
knn.fit(X_train, y_train)

# Evaluate the classifier using cross-validation
cv_scores = cross_val_score(knn, X, y, cv=20)  # 20-fold cross-validation
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV accuracy: {cv_scores.mean()}')

# Test the classifier
test_accuracy = knn.score(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Plot accuracy for different values of k (1 to 20)
k_values = range(1, 21)
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))

# Plotting the accuracies
plt.plot(k_values, train_accuracies, label='Training Accuracy')
plt.plot(k_values, test_accuracies, label='Test Accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different K Values')
plt.legend()
plt.show()

# Using the trained model for predictions
def predict_bill(dimensions):
    bill_data = pd.DataFrame([dimensions], columns=X.columns)  # Adjust for actual columns
    prediction = knn.predict(bill_data)
    prediction_label = 'Genuine' if prediction == 1 else 'Fake'
    return prediction_label

# Example input for the classifier
example_bill = [30.5, 12.5, 12.0, 0.5, 0.5, 6.5]  #Example dimensions: [diagonal, height_left, height_right, margin_low, margin_upper, length]
print(predict_bill(example_bill))
