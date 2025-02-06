import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load the csv file
df = pd.read_csv("iris.csv")

print(df.head())

# Select independent and dependent variable
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

# # Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# Define base classifiers
rf_clf = RandomForestClassifier(random_state=42)
log_clf = LogisticRegression(max_iter=200)
svc_clf = SVC(probability=True)

voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('log', log_clf), ('svc', svc_clf)],
    voting='soft'  # 'soft' for probability-based voting, 'hard' for majority voting
)

# Train the ensemble model
voting_clf.fit(X_train, y_train)

# Save the trained model using pickle
with open('iris_voting_model.pkl', 'wb') as file:
    pickle.dump(voting_clf, file)

# Feature scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)


