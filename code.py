# importing libaries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import validation_curve

#creating a design profile
import seaborn as sns
sns.set(rc={'figure.figsize': [10, 10]}, font_scale=1.2)

#filtering to ignore warnings
import warnings
warnings.filterwarnings('ignore')


# import dataset
dataset = pd.read_csv('train.csv')


# define dependent and independent variables
X = dataset[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt',
      'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']]
y = dataset['price_range']



# splitting the data into training and testing 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, random_state=42)



# functions for initial investigation - please run using spyder

def analyse_data():
      path = "."  #absolute or relative path to the folder containing the file. 
                #"." for current folder
    
      filename_read = os.path.join(path, "train.csv")
      df = pd.read_csv(filename_read) # for analysis
      
      # dataset info
      print  ("\nDataset information: \n")
      print(df.info())

      '''Start of analysis of data'''
      # Strip non-numerics
      df = df.select_dtypes(include=['int', 'float'])
      
      print  ("\nCounting of data: \n")
      
      #details on the data in numbers 
      for col in df.columns:
          print(col + '\n_____________')
          print(df[col].value_counts())
          print('_____________________________\n')

      
      print("\nStatistics: \n")

      #display statistics 
      get_stats(df)

      #visualise data so that it is clear what the data is showing
      df.hist(figsize=(20,20))
      plt.show()
      #sns.pairplot(df,hue='price_range')

      #heat map isa correlation matrix used to show the correlations between each field 
      plt.figure(figsize=(16,16))
      sns.heatmap(df.corr(), annot=True, fmt=".2f")
      '''End of analysis of data'''
      
      
def get_stats(df):
    
      headers = list(df.columns.values)
      fields = []
      # Perform basic statistics (mean, variance, standard deviation, z scores) on a dataframe.
      for field in headers:
          fields.append({
              'name' : field,
              'mean': df[field].mean(),
              'var': df[field].var(),
              'sdev': df[field].std(),# how dispersed the data is in relation to the mean
          
          })
          
          
      #display statistics 
      for field in fields:
          print(field)
          

def valC(model, paramName, interval, range):                       # Created specific parameters to be changed for each model

      # Range for the parameter (from 1 to range)
      paramRange = np.arange(1, range, 1)

      # Calculate accuracy on training and test set using the parameter with 10-fold cross validation, parameter and model can be changed by the user. 
      train_score, test_score = validation_curve(model(), X_train, y_train, param_name = paramName, param_range = paramRange, cv = interval, scoring = "accuracy")
      

      # Calculating mean and standard deviation of training and testing score
      trainScore_mean = np.mean(train_score, axis = 1)
      std_train_score = np.std(train_score, axis = 1)
      testScore_mean = np.mean(test_score, axis = 1)
      std_test_score = np.std(test_score, axis = 1)
      

     
      # Plotting mean accuracy scores for training and testing scores
      plt.plot(paramRange, trainScore_mean,
      label = "Training Score", color = 'blue')
      plt.plot(paramRange, testScore_mean,
      label = "Cross Validation Score", color = 'green')
      
       # Plotting graph elements
      plt.title("Validation Curve")
      plt.xlabel(paramName)
      plt.ylabel("Accuracy")
      plt.tight_layout()
      plt.legend(loc = 'best')
      plt.show()



      
#functions for models, for ease of access

def logistic():
    
      print ("\nLogistic Regression:\n")

      # building the model
      logisticModel = LogisticRegression(max_iter = 10000)
      logisticModel.fit(X_train, y_train)

      # making our predicitons
      y_pred = logisticModel.predict(X_test)
      
      #cross-validation
      scores = cross_val_score(logisticModel, X, y, cv=10)
      print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

      #prints accuracy using metrics
      print("\n (metric) Accuracy:", metrics.accuracy_score(y_test, y_pred), "\n")

      # printing the classification report
      print(classification_report(y_test, y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(logisticModel, X_test, y_test)  
      plt.title('Logistic Regression')
      plt.show()


      #reference: labs


def decisionTree():
        

      print("\nDecision Tree:\n")

      # building the model
      clf = DecisionTreeClassifier(max_depth = 7)                         #max_features=20
      clf = clf.fit(X_train,y_train)

      # making the prediction
      y_pred = clf.predict(X_test)
     
      #cross-validation
      scores = cross_val_score(clf, X, y, cv=10)
      print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

      #prints accuracy using metrics
      print("\n (metric) Accuracy:", metrics.accuracy_score(y_test, y_pred), "\n")

      # printing the classification report
      print(classification_report(y_test, y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(clf, X_test, y_test) 
      plt.title('Decision Tree')
      plt.show()
      
      #reference: https://stackabuse.com/decision-trees-in-python-with-scikit-learn/


def SVM():
        
      print("\nSVM:\n")

      # building the model
      sv  = SVC(kernel='rbf', C=240, random_state = 42)
      sv.fit(X_train,y_train)
      
      #making our prediction
      y_pred = sv.predict(X_test)
      
      #cross-validation
      scores = cross_val_score(sv, X, y, cv=10)
      print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

      #prints accuracy using metrics
      print("\n (metric) Accuracy: ", metrics.accuracy_score(y_test, y_pred), "\n")

      # printing the classification report
      print(classification_report(y_test, y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(sv, X_test, y_test)  
      plt.title('SVM')
      plt.show()
      

      #ref: https://analyticsindiamag.com/understanding-the-basics-of-svm-with-example-and-python-implementation/
      

def nB():
       
      print("\nNaive Bayes:\n")

      # building the model
      nb = GaussianNB()
      nb.fit(X_train,y_train)
      
      # making predictions
      y_pred = nb.predict(X_test)
      
      #cross-validation
      scores = cross_val_score(nb, X, y, cv=10)
      print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

      #prints accuracy using metrics
      print("\n (metric) Accuracy: ", metrics.accuracy_score(y_test, y_pred), "\n")
      
      # printing the classification report 
      print(classification_report(y_test,y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(nb, X_test,y_pred)
      plt.title('Naive Bayes')
      plt.show()
      
      


def kNear():

      print("\nK-neighbors:")

      # building the model
      neigh = KNeighborsClassifier(n_neighbors=9)  #using the validation curve - we found the higher number of neighbors, the higher the accuracy. 3 neighbors gave 93% and 9 neighbors gave 95% accuracy.
      neigh.fit(X_train, y_train)

      # making predictions
      y_pred = neigh.predict(X_test)
      
      #cross-validation
      scores = cross_val_score(neigh, X, y, cv=10)
      print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

      #prints accuracy using metrics
      print("\n (metric) Accuracy: ", metrics.accuracy_score(y_test, y_pred), "\n")
      
      # printing the classification report 
      print(classification_report(y_test,y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(neigh, X_test,y_pred)
      plt.title('KNN')
      plt.show() 




#analyse_data()


#logistic()             #70%  (rounded)     #after changing training size from 0.8 to 0.6 now 73% from 70%
#decisionTree()         #83%  (rounded)     #after changing max_depth from n/a to 7, increase from 80% to 83%
#SVM()                  #95%  (rounded)     #after changing the C value from 1 to 240 now 97% from 95% 
#nB()                   #81%  (rounded)      
#kNear()                #91%  (rounded)     #after changing the no. neighbors from 3 to 20, accuracy increased from 90% to 93%



#valC(KNeighborsClassifier, "n_neighbors", 10, 20)                 #how to use: val(modelType, "parameter", folds, range)
#valC(DecisionTreeClassifier, "max_features", 10, 35)               #there are a maximum of 20 features 
#valC(DecisionTreeClassifier, "max_depth", 10, 20)       
#valC(SVC, "C", 10, 500)             # after changing the C value from 1 to 240 now 97% from 95%  
#valC(SVC, "max_iter", 10, 500)      # shows that around 86 to 94 is the optinmal result (increment is too small)
#valC(LogisticRegression, "max_iter", 10, 500)
#valC(GaussianNB, "var_smoothing", 10, 0.000000001) #to test, this we had to change the minimum values seen in val(). 



'''

README

To use the program - un comment specific lines to run specific functions. 
breakdown: 

analyse_data() --> code used for initial investigation

logistic()
decisionTree()
...
...         --> functions to run each model



valC(KNeighborsClassifier, "n_neighbors", 10, 100)  
... 
...         --> specific functions for each model, to run validation curves.   
                how to use: valC(modelType, "parameter", folds, range)

'''