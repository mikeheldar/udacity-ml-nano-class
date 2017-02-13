import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
#dataframe = pd.read_fwf('brain_body.txt')
#x_values = dataframe[['Brain']]
#y_values = dataframe[['Body']]

#train
#body_reg = linear_model.LinearRegression()
#body_reg.fit(x_values,y_values)

#vis
#plt.scatter(x_values,y_values)
#plt.plot(x_values, bodey_reg.predict(x_values))

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model

bmi_life_model = linear_model.LinearRegression()
bmi_life_model.fit(x_values,y_values)


# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)

print'laos_life_exp: ', laos_life_exp[0][0]
