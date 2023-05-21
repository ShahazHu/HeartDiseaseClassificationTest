# HeartDiseaseClassificationTest
The project involves using AI in healthcare applications. Given a dataset containing information of patients attribute information involving their heart, the objective is to create a classifier model that will best predict whether the patient will develop heart disease.

## Evaluation
Testing the first model, we got an accuracy of 0.7912 and an AUC score of 0.7859. The next goal was to test different parameters and see which one would perform the best. The following list of parameters were used to test the model:
```
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
```


## Conclusion
With our first run through, we see that combination 10 and 11 performed well. Accuracy isn’t the best way to determine whether a model has performed well, but 10 and 11 had the highest accuracies and highest AUC score as well. The parameters used to achieve this were, for combination 10: {'layer1': 64, 'layer2': 32, 'activation': 'relu', 'optimizer': 'sgd', 'epochs': 100}, and model 11: {'layer1': 64, 'layer2': 32, 'activation': 'relu', 'optimizer': 'sgd', 'epochs': 100}. Since they were close, we can use other factors such as precision, recall, F-measure, error rate.
```
Classification report:
precision recall f1-score support
Negative for Heart Disease 0.85 0.80 0.83 41
Positive for Heart Disease 0.85 0.88 0.86 50
accuracy 0.85 91
macro avg 0.85 0.84 0.84 91
weighted avg 0.85 0.85 0.85 91
```
Coincidentally, both performed the exact same. Running and testing the combinations again then gave us different results:
![image](https://github.com/ShahazHu/HeartDiseaseClassificationTest/assets/61039853/e8154f74-28cd-4cc2-b354-3bb6ed43ef57)
![image](https://github.com/ShahazHu/HeartDiseaseClassificationTest/assets/61039853/e6f25989-473d-4a76-b224-89a9c25ba2be)
Now with the newly trained model we see that 10 and 1 are nearly identical, using the information provided from the classification report function, we can break that tiebreaker.
![image](https://github.com/ShahazHu/HeartDiseaseClassificationTest/assets/61039853/3bd96a35-aa29-47ae-8625-9159294bc1c6)
![image](https://github.com/ShahazHu/HeartDiseaseClassificationTest/assets/61039853/f25ed07e-eee4-4c2a-8cb1-3aefc06f40f6)
Parameter combination 1 has a slightly higher AUC of 0.8390 compared to the second model which has an AUC of 0.8324. Therefore, the first model performed better in terms of AUC.

Looking at the classification reports, the first model has slightly higher precision for the positive class (presence of heart disease) compared to the second model, while the second model has slightly higher recall for the positive class. However, the overall F1 score is similar for both models, indicating a similar balance between precision and recall.

Overall, Parameter combination 1 performed better in terms of AUC, while the two models have similar performance in terms of accuracy and overall F1 score.

The best performing parameter combination changes every time it was ran because the model training process is non-deterministic The training process also involves a certain level of randomness We may have to make the model with more neural layers and run the model multiple times and average the result because of this randomness in order to determine which set of parameters is actually the best

