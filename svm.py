#-------------------------------------------------------------------------
# AUTHOR: Wan Suk Lim
# FILENAME: svm.py
# SPECIFICATION: svm
# FOR: CS 4210- Assignment #3
# TIME SPENT: 30 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv

dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1])

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

for ci in c: #iterates over c
    for degreei in degree: #iterates over degree
        for kerneli in kernel: #iterates kernel
           for decisioni in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=ci, degree=degreei, kernel=kerneli, decision_function_shape=decisioni)

                #Fit SVM to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                #--> add your Python code here
                rightPrediction = 0
                for testSample in dbTest:
                    class_predicted = clf.predict([testSample[:-1]])[0]
                    if class_predicted == testSample[-1]:
                        rightPrediction += 1

                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                accuracy = rightPrediction/len(dbTest)
                if accuracy > highestAccuracy:
                    highestAccuracy = accuracy
                    print("Highest SVM accuracy so far: " + str(highestAccuracy) + ", Parameters: c=" + str(ci) + ", degree=" + str(degreei) + ", kernel=" + kerneli + ", decision_function_shape=" + decisioni)











