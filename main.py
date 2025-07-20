import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error,mean_squared_error 

#Import data as a Pandas DataFrame
insects = pd.read_csv('./data/insects.csv', sep='\t')
insects = pd.DataFrame({
    'continent': insects['continent'],
    'latitude': insects['latitude'],
    'sex': insects['sex'],
    'wingsize': insects['wingsize']
})

# Filter the data to only male insects
insects = insects[insects.sex == 1]

# Features variable
X_insects = insects[['wingsize']]
# Target variable
y_insects = insects['latitude']

insects.head()

#Plot the data
plt.scatter(X_insects, y_insects, label="Actual Data", color='g')
plt.xlabel("Wing size")
plt.ylabel("Latitude")
plt.legend()
plt.show()

insects_regression = LinearRegression()

insects_regression.fit(X_insects, y_insects)

# Predict the target for the whole dataset
latitude_predictions = insects_regression.predict(X_insects)

#Predict the target for a new data point
new_insect = pd.DataFrame({
    'wingsize': [800]
})
new_insect['latitude'] = insects_regression.predict(new_insect)
print(f"New insect is:\n{new_insect}")

#Plot the predictions compared to the actual data
plt.scatter(X_insects, y_insects, label="Actual Data", color='g')
plt.scatter(X_insects, latitude_predictions, label="Predicted Data", c='r')
plt.xlabel("Wing size")
plt.ylabel("Latitude")
plt.legend()
plt.show()

#Get Evalutative Data from the model
print(f"Model coefficient :{insects_regression.coef_}")
print(f"Model y intercept :{insects_regression.intercept_}")
print(f"Model score :{insects_regression.score(X_insects,y_insects)}")
mae = mean_absolute_error(y_true=y_insects,y_pred=latitude_predictions) 
mse = mean_squared_error(y_true=y_insects,y_pred=latitude_predictions)
print("MAE:",mae) 
print("MSE:",mse)