# importing the required libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Reading the data
data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')

# Check if there any null value in the Dataset
if data.isnull != True:
    # Lets scatter plot between the 'Marks Percentage' and 'Hours Studied'
    sns.set_style('darkgrid')
    sns.scatterplot(y=data['Scores'], x=data['Hours'])
    plt.title('Marks Vs Study Hours', size=20)
    plt.ylabel('Marks Percentage', size=12)
    plt.xlabel('Hours Studied', size=12)
    plt.show()

    # Lets plot a regression line to confirm the correlation.

    sns.regplot(x=data['Hours'], y=data['Scores'])
    plt.title('Regression Plot', size=20)
    plt.ylabel('Marks Percentage', size=12)
    plt.xlabel('Hours Studied', size=12)
    plt.show()

    print(data.corr())

    # Training the Model

    # 1) Splitting the Data

    # Defining X and y from the Data

    X = data.iloc[:, :-1].values
    y = data.iloc[:, 1].values

    # Spliting the Data in two
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    # 2) Fitting the Data into the model

    regression = LinearRegression()
    regression.fit(train_X, train_y)
    print("---------Model Trained---------")

    # Predicting the Percentage of Marks
    pred_y = regression.predict(val_X)
    prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
    print(prediction)

    # Comparing the Predicted Marks with the Actual Marks
    compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
    print(compare_scores)

    # Visually Comparing the Predicted Marks with the Actual Marks

    plt.scatter(x=val_X, y=val_y, color='blue')
    plt.plot(val_X, pred_y, color='Black')
    plt.title('Actual vs Predicted', size=20)
    plt.ylabel('Marks Percentage', size=12)
    plt.xlabel('Hours Studied', size=12)
    plt.show()

    # Evaluating the Model
    # Calculating the accuracy of the model
    print('Mean absolute error: ', mean_absolute_error(val_y, pred_y))

    '''Small value of Mean absolute error states
     that the chances of error or wrong forecasting through the model are very less.'''

    # Lets predict the marks for students based on student study hours.
    hours = [float(input("Enter No.Of Hours: "))]
    answer = regression.predict([hours])
    print("Score = {}".format(round(answer[0], 3)))
else:
    print("There is a null value in the data set")
    pass