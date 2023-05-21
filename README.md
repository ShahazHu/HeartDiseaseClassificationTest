# HeartDiseaseClassificationTest
The project involves using AI in healthcare applications. Given a dataset containing information of patients attribute information involving their heart, the objective is to create a classifier model that will best predict whether the patient will develop heart disease.

## Evaluation
Testing the first model, we got an accuracy of 0.7912 and an AUC score of 0.7859. The next goal was to test different parameters and see which one would perform the best. The following list of parameters were used to test the model:
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
{'layer1': 64, 'layer2': 32, 'activation': 'relu', 'optimizer': 'sgd', 'epochs': 100}



## Conclusion
With our first run through, we see that combination 10 and 11 performed well. Accuracy isn’t the best way to determine whether a model has performed well, but 10 and 11 had the highest accuracies and highest AUC score as well. The parameters used to achieve this were, for combination 10: {'layer1': 64, 'layer2': 32, 'activation': 'relu', 'optimizer': 'sgd', 'epochs': 100}, and model 11: {'layer1': 64, 'layer2': 32, 'activation': 'relu', 'optimizer': 'sgd', 'epochs': 100}. Since they were close, we can use other factors such as precision, recall, F-measure, error rate.
Parameter combination 10: {'layer1': 32, 'layer2': 64, 'activation': 'relu', 'optimizer': 'sgd', 'epochs': 100}
Accuracy: 0.8462
AUC: 0.8424
Classification report:
precision recall f1-score support
Negative for Heart Disease 0.85 0.80 0.83 41
Positive for Heart Disease 0.85 0.88 0.86 50
accuracy 0.85 91
macro avg 0.85 0.84 0.84 91
weighted avg 0.85 0.85 0.85 91
