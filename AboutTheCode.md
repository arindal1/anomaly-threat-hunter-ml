# Let's Break down the Code structure:

### AnomalyDetection.ipynb

```python
import numpy as np
import pandas as pd
```
- `import numpy as np` imports the NumPy library as np, providing support for numerical operations.
- `import pandas as pd` imports the Pandas library as pd, allowing data manipulation and analysis using dataframes.

```python
df = pd.read_csv('NSL-KDD/KDDTest+.txt', delimiter=',')
```
- `pd.read_csv('NSL-KDD/KDDTest+.txt', delimiter=',')` reads a CSV file named 'KDDTest+.txt' with ',' as the delimiter and assigns it to a Pandas dataframe called 'df'.

```python
df.sample(5)
```
- `df.sample(5)` displays a random sample of 5 rows from the dataframe 'df'.

```python
df.shape
```
- `df.shape` returns the number of rows and columns in the dataframe, providing the shape of the dataset.

```python
df.info()
```
- `df.info()` displays concise information about the dataframe, including the data types and non-null counts of each column.

```python
column_names = [...]
```
- `column_names` is a list containing the names of columns you want to assign to the dataframe.

```python
df.columns = column_names
```
- `df.columns = column_names` sets the column names of the dataframe 'df' to the names specified in `column_names`.

```python
categorical_cols = ['protocol_type', 'service', 'flag', 'neptune']
df = pd.get_dummies(df, columns=categorical_cols)
```
- `categorical_cols` lists the categorical columns.
- `pd.get_dummies(df, columns=categorical_cols)` creates dummy variables for the categorical columns specified in `categorical_cols` and replaces them in the dataframe.

```python
numeric_cols = ['duration', 'src_bytes', ...]
```
- `numeric_cols` is a list containing the names of numeric columns.

```python
from sklearn.preprocessing import MinMaxScaler
```
- `from sklearn.preprocessing import MinMaxScaler` imports the MinMaxScaler from scikit-learn for feature scaling.

```python
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```
- `scaler = MinMaxScaler()` creates an instance of MinMaxScaler.
- `scaler.fit_transform(df[numeric_cols])` scales the numeric columns specified in `numeric_cols` using Min-Max scaling.

```python
df['target'] = df['target'].apply(lambda x: 1 if x != 21 else 0)
```
- Modifies the 'target' column, assigning 1 to values different from 21 (indicating attacks) and 0 to 21 (indicating normal).

This covers the initial data loading, preprocessing, and feature engineering steps. If you'd like a detailed explanation of the rest of the code, please let me know!

```python
import matplotlib.pyplot as plt
import seaborn as sns
```
- Imports the Matplotlib library for plotting as `plt` and the Seaborn library for statistical data visualization as `sns`.

```python
df2.describe()
```
- Calls the `describe()` function on `df2`, which provides summary statistics of the dataframe.

```python
df2['target'].value_counts()
```
- Computes the count of each unique value in the 'target' column of `df2`.

```python
plt.pie(df2['target'].value_counts(), labels=['attacks','normal'], autopct="%0.2f")
plt.show()
```
- Plots a pie chart to visualize the distribution of 'target' values using Matplotlib.

```python
plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
```
- Creates a heatmap of the correlation matrix of the dataframe using Seaborn and Matplotlib.

```python
sns.pairplot(df.sample(1000), vars=numeric_cols[:3], hue='target')
plt.suptitle('Pairplot of Numerical Features (Sampled)')
plt.show()
```
- Generates a pairplot for a sample of the dataframe, focusing on the first three numerical features, with color encoding by the 'target' variable.

```python
plt.figure(figsize=(18, 12))
for i, col in enumerate(numeric_cols[:4]):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()
```
- Creates a grid of histograms to display the distribution of the first four numerical features.

```python
plt.figure(figsize=(18, 12))
for i, col in enumerate(numeric_cols[:4]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='target', y=col, data=df)
    plt.title(f'Boxplot of {col} by target')
plt.tight_layout()
plt.show()
```
- Generates boxplots to visualize the distribution of numerical features by the 'target' variable.

```python
plt.figure(figsize=(18, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=col, data=df, hue='target')
    plt.title(f'Countplot of {col} by target')
plt.tight_layout()
plt.show()
```
- Plots countplots to display the count of each categorical feature by the 'target' variable.

These sections cover data visualization, which is crucial for understanding the dataset and making informed decisions about further analysis and modeling. If you have more specific questions or if you'd like to continue with the next parts of the code, feel free to ask!

```python
# Create binary features for each protocol type
df2['is_protocol_tcp'] = (df2['protocol_type_tcp'] == 1).astype(int)
df2['is_protocol_udp'] = (df2['protocol_type_udp'] == 1).astype(int)
df2['is_protocol_icmp'] = (df2['protocol_type_icmp'] == 1).astype(int)
```
- Creates binary features for each protocol type (TCP, UDP, ICMP) indicating their presence with 1 or 0.

```python
# Create binary features for each flag
df2['is_flag_SF'] = (df2['flag_SF'] == 1).astype(int)
df2['is_flag_S0'] = (df2['flag_S0'] == 1).astype(int)
df2['is_flag_REJ'] = (df2['flag_REJ'] == 1).astype(int)
```
- Creates binary features for each flag (SF, S0, REJ) indicating their presence with 1 or 0.

```python
# Aggregate services into categories
web_services = ['http', 'https', 'smtp', 'ftp', 'ssh', 'ssl']
df2['service_category'] = df['service'].apply(lambda x: 'Web' if x in web_services else 'Others')
```
- Groups services into categories, classifying specific services as 'Web' services or 'Others'.

```python
df2.head
```
- Attempts to access the `head` attribute of `df2` but doesn't actually call the function. It should be `df2.head()` to display the first few rows of the DataFrame.

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

random_model = RandomForestClassifier(random_state=42)

n_features_to_select = 10  # Adjust this based on your preference
rfe = RFE(estimator=rf_model, n_features_to_select=n_features_to_select)
fit = rfe.fit(X_train_encoded, y_train)

selected_features = [X_train_encoded.columns[i] for i in range(len(rfe.support_)) if rfe.support_[i]]
print("Selected Features:", selected_features)
```
- Imports necessary libraries and performs Recursive Feature Elimination (RFE) using a RandomForestClassifier to select a specified number of features based on their importance.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

# ... (omitting repetitive code)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Imports necessary libraries for model training and evaluation, and then splits the data into training and testing sets.

```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_encoded, y_train)

svm_model = SVC(random_state=42)
svm_model.fit(X_train_encoded, y_train)

log_reg_model = LogisticRegression(random_state=42, max_iter=10000)  # Increase max_iter
log_reg_model.fit(X_train_encoded, y_train)

nn_model = MLPClassifier(random_state=42)
nn_model.fit(X_train_encoded, y_train)
```
- Initializes and trains machine learning models: RandomForestClassifier, SVC (Support Vector Classifier), LogisticRegression, and MLPClassifier (Neural Network).

```python
# ... (omitting code for making predictions and evaluating models)

X_train_selected = X_train_encoded[selected_features]
X_test_selected = X_test_encoded[selected_features]

rf_model_selected = RandomForestClassifier(random_state=42)
rf_model_selected.fit(X_train_selected, y_train)

rf_predictions_selected = rf_model_selected.predict(X_test_selected)

rf_accuracy_selected = accuracy_score(y_test, rf_predictions_selected)
print("Accuracy Score with Selected Features:", rf_accuracy_selected)
print("Classification Report:")
print(classification_report(y_test, rf_predictions_selected), "\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions_selected))
```
- Performs training and evaluation with a subset of selected features using a RandomForestClassifier.

The remaining code involves hyperparameter tuning, creating an ensemble model (VotingClassifier), and evaluating the ensemble model's performance.

```python
from sklearn.model_selection import GridSearchCV

# ... (omitting repetitive code for defining parameter grids)

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=rf_params, cv=3)
svm_grid = GridSearchCV(SVC(random_state=42), param_grid=svm_params, cv=3)
log_reg_grid = GridSearchCV(LogisticRegression(random_state=42), param_grid=log_reg_params, cv=3)
nn_grid = GridSearchCV(MLPClassifier(random_state=42), param_grid=nn_params, cv=3)

rf_grid.fit(X_train_encoded, y_train)
svm_grid.fit(X_train_encoded, y_train)
log_reg_grid.fit(X_train_encoded, y_train)
nn_grid.fit(X_train_encoded, y_train)
```
- Imports GridSearchCV for hyperparameter tuning, defines parameter grids for each model, and performs a grid search to find the best parameters for each model.

```python
print("Random Forest - Best Parameters:", rf_grid.best_params_)
print("SVM - Best Parameters:", svm_grid.best_params_)
print("Logistic Regression - Best Parameters:", log_reg_grid.best_params_)
print("Neural Network - Best Parameters:", nn_grid.best_params_)
```
- Prints the best hyperparameters found for each model during the grid search.

```python
from sklearn.ensemble import VotingClassifier

models = [('Random Forest', rf_model), ('SVM', svm_model), ('Logistic Regression', log_reg_model), ('Neural Network', nn_model)]

voting_clf = VotingClassifier(estimators=models, voting='hard')

voting_clf.fit(X_train_encoded, y_train)
```
- Imports VotingClassifier, creates a list of models and their names, then creates a VotingClassifier with a 'hard' voting strategy and fits it on the training data.

```python
ensemble_predictions = voting_clf.predict(X_test_encoded)

ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

print("Ensemble (Voting Classifier) Accuracy:", ensemble_accuracy)
print("Classification Report for Ensemble:")
print(classification_report(y_test, ensemble_predictions))
print("Confusion Matrix for Ensemble:")
print(confusion_matrix(y_test, ensemble_predictions))
```
- Makes predictions using the ensemble (VotingClassifier) and evaluates its accuracy, classification report, and confusion matrix.



### app.py


1. **Importing Necessary Libraries:**
   ```python
   import streamlit as st
   import pickle
   import numpy as np
   ```

   - `streamlit` is a Python library used to create interactive web applications for data science and machine learning projects.
   - `pickle` is used for serializing and deserializing Python objects, in this case, to load the pre-trained machine learning model.
   - `numpy` is a popular library for numerical computing in Python.

2. **Loading the Random Forest Model:**
   ```python
   with open('selected_randomforest.pkl', 'rb') as file:
       model = pickle.load(file)
   ```

   - The code loads a pre-trained Random Forest model using `pickle`. The model is stored in a file named 'selected_randomforest.pkl'.

3. **Defining Selected Features:**
   ```python
   selected_features = ['dst_bytes', 'unknown_feature', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_srv_serror_rate', 'service_http', 'service_private', 'neptune_neptune', 'neptune_normal']
   ```

   - This list contains the features that were used to train the Random Forest model.

4. **Creating the Streamlit Web App:**
   ```python
   st.title('Intrusion Detection System')
   ```

   - Sets the title of the web application to "Intrusion Detection System".

5. **Getting User Input for Selected Features:**
   ```python
   for feature in selected_features:
       user_input[feature] = st.slider(f'Enter {feature}', min_value=0.0, max_value=1.0, step=0.01)
   ```

   - This section uses Streamlit's `st.slider` to create sliders for the user to input values for each selected feature.

6. **Making Predictions:**
   ```python
   if st.button('Predict'):
       input_array = np.array([user_input[feature] for feature in selected_features]).reshape(1, -1)
       prediction = model.predict(input_array)
   ```

   - If the user clicks the "Predict" button, it collects the user's input and uses the pre-loaded model to predict whether it's normal traffic or an anomaly.

7. **Displaying Prediction Results:**
   ```python
   if prediction == 0:
       st.success('Prediction: Normal traffic (class 0)')
   else:
       st.error('Prediction: Anomaly detected (class 1)')
   ```

   - Displays the prediction result based on the model's prediction. If the prediction is 0, it's classified as normal traffic. If the prediction is 1, it's classified as an anomaly.

The Streamlit app provides a simple interface for a user to input values for selected features and then predicts whether the input represents normal traffic or an anomaly based on the pre-trained Random Forest model.
