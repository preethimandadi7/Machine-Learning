# Importing the librarys 
#numpy is used to made the operations on the datasets
import numpy as np 
#pandas library used to made the highperformence data filtering and data manupulations on the datasets
import pandas as pd

# Importing sklearn machine learning models and the operations 

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# importing matplot lib to plot the charts
import matplotlib.pyplot as plt


# Reading the data from the local folder

df = pd.read_csv('house_data.csv')

# filtering the data removing NULL, null,NaN and empty data
df = df.dropna()


#Method returns the adjusting value of R-squared Mean
def adjR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)

# Splitting the exsiting data into training and test sets of data 80% is taken in train data
train_data,test_data = train_test_split(df,train_size = 0.8,random_state=3)

#Loading the Linear ergression model from the scikit

linearRegression = linear_model.LinearRegression()

# Creating x and y data for predictions
X_train = np.array(train_data['sqft_living'], dtype=pd.Series).reshape(-1,1)
y_train = np.array(train_data['price'], dtype=pd.Series)
# fitting the data to model
linearRegression.fit(X_train,y_train)

# Reshapping the data
X_test = np.array(test_data['sqft_living'], dtype=pd.Series).reshape(-1,1)
y_test = np.array(test_data['price'], dtype=pd.Series)

# Predicting the data
pred = linearRegression.predict(X_test)

# Finding the Mean Square values
rmsesm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rtrsm = float(format(linearRegression.score(X_train, y_train),'.3f'))
rtesm = float(format(linearRegression.score(X_test, y_test),'.3f'))
cv = float(format(cross_val_score(linearRegression,df[['sqft_living']],df['price'],cv=5).mean(),'.3f'))

# Printin the stats
print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))
print('Intercept: {}'.format(linearRegression.intercept_))
print('Coefficient: {}'.format(linearRegression.coef_))

# Creating the frame 
evaluation = pd.DataFrame({'Model': [],'Details':[],'Root Mean Squared Error (RMSE)':[],'R-squared (training)':[],'Adjusted R-squared (training)':[],'R-squared (test)':[],'Adjusted R-squared (test)':[],'5-Fold Cross Validation':[]})
r = evaluation.shape[0]
evaluation.loc[r] = ['Simple Linear Regression','-',rmsesm,rtrsm,'-',rtesm,'-',cv]
print(evaluation)

# Showing the graph of the predictions

plt.figure(figsize=(6.5,5))
plt.scatter(X_test,y_test,color='green',label="data", alpha=.1)
plt.plot(X_test,pred,color="red",label="Predicted Linear Regression")
plt.xlabel("Living Space (sqft)", fontsize=6)
plt.ylabel("Price ($)", fontsize=6)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()