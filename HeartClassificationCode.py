import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

np.random.seed(23)


#load the dataframe from the CSV file
data = pd.read_csv('/content/heart.csv')

#prints the dimensions of the data
print(f"Dimensions of dataframe: {data.shape}")

#print columns 
print('\033[1m' + "\nColumns:" + '\033[0m')
for i, params in enumerate(data.columns):
  print(f"{params}: {data[params].describe()}" )
  print("\n")

#looking at the top 5 rows 
print(data.head())

#split data into x for features and y for target
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


#scale the features using StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)


#calculate the mutual information scores
info_scores = mutual_info_classif(x, y)

print("\n --")

#convert x to a DataFrame and print the feature names and their information gain scores
x_df = pd.DataFrame(x, columns=data.columns[:-1])
for i, feature in enumerate(x_df.columns):
    print(f"{feature}: {info_scores[i]}")


#[lot the feature importance scores
plt.barh(x_df.columns, info_scores)
plt.title('Information Gain Score')
plt.show()

print("\n --")

#train a random forest classifier to compute feature importance
model = RandomForestClassifier()
model.fit(x, y)

#get feature importance scores
importance_scores = model.feature_importances_

#print the feature importance scores in descending order
importance_df = pd.DataFrame({'Features': x_df.columns, 'Importance': importance_scores})
print(importance_df.sort_values(by='Importance'))

#plot the feature importance scores
plt.barh(x_df.columns, importance_scores)
plt.title('Feature Importance Scores')
plt.show()

#split dataset to train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


#define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_dim=13),
    tf.keras.layers.Dense(64, activation='relu', input_dim=13),
    tf.keras.layers.Dense(1, activation='sigmoid', input_dim=13)
])

#compile and train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

#evaluate the model on the testing set
y_pred = model.predict(x_test)
y_pred = np.round(y_pred).flatten()

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

#create a function that creates a model and can use different parameters
def neuralNetworkTest(params):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(params['layer1'], activation=params['activation'], input_dim=13),
        tf.keras.layers.Dense(params['layer2'], activation=params['activation'], input_dim=13),
        tf.keras.layers.Dense(1, activation='sigmoid', input_dim=13)
    ])

    model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=params['epochs'], batch_size=32, verbose=0)

    #evaluate the model
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Negative for Heart Disease", "Positive for Heart Disease"])


    return accuracy, auc, report

#list of parameters to test on the model
params_list = [
    {'layer1': 32, 'layer2': 64, 'activation': 'relu', 'optimizer': 'adam', 'epochs': 100},
    {'layer1': 64, 'layer2': 32, 'activation': 'relu', 'optimizer': 'adam', 'epochs': 100},
    {'layer1': 32, 'layer2': 64, 'activation': 'relu', 'optimizer': 'adam', 'epochs': 50},
    {'layer1': 64, 'layer2': 64, 'activation': 'relu', 'optimizer': 'adam', 'epochs': 100},
    {'layer1': 32, 'layer2': 64, 'activation': 'relu', 'optimizer': 'adam', 'epochs': 150},
    {'layer1': 32, 'layer2': 64, 'activation': 'sigmoid', 'optimizer': 'adam', 'epochs': 100},
    {'layer1': 64, 'layer2': 32, 'activation': 'sigmoid', 'optimizer': 'adam', 'epochs': 100},
    {'layer1': 32, 'layer2': 64, 'activation': 'sigmoid', 'optimizer': 'sgd', 'epochs': 100},
    {'layer1': 64, 'layer2': 32, 'activation': 'sigmoid', 'optimizer': 'sgd', 'epochs': 100},
    {'layer1': 32, 'layer2': 64, 'activation': 'relu', 'optimizer': 'sgd', 'epochs': 100},
    {'layer1': 64, 'layer2': 32, 'activation': 'relu', 'optimizer': 'sgd', 'epochs': 100},
    
]

#evaluate each parameter
accuracies = []
AUC_score = []
for i, params in enumerate(params_list):
  accuracy, auc, report = neuralNetworkTest(params)
  accuracies.append(accuracy)
  AUC_score.append(auc)
  print(f"Parameter combination {i+1}: {params}")
  print(f"Accuracy: {accuracy:.4f}")
  print(f"AUC: {auc:.4f}")
  print(f"Classification report:\n{report}")

#creating a bar graph with the outputted auc and accuracy score
x_ticks = [f"{i+1}" for i in range(len(params_list))]
plt.bar(x_ticks, accuracies)
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Accuracy Score Comparison')

for i, v in enumerate(accuracies):
    plt.text(i, v+0.01, f"{v:.4f}", ha='center', fontsize=10)

plt.show()

print('\n')

#creating a bar graph with the outputted auc and accuracy score
x_ticks = [f"{i+1}" for i in range(len(params_list))]
plt.bar(x_ticks, AUC_score)
plt.ylim([0, 1])
plt.ylabel('AUC')
plt.xlabel('Model')
plt.title('AUC Score Comparison')

for i, v in enumerate(AUC_score):
    plt.text(i, v+0.01, f"{v:.4f}", ha='center', fontsize=10)

plt.show()
