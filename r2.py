#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 
import warnings
warnings.filterwarnings("ignore")


# In[2]:


dataTrain = pd.read_csv(r'C:\Users\almon\Desktop\house_data.csv')
dataTrain.head()


# In[3]:


type(dataTrain)


# In[4]:


dataTrain.shape


# In[5]:


dataTrain.isnull().sum()


# In[6]:


dataTrain.shape


# In[7]:


dataTrain.dtypes


# In[8]:


plt.figure(figsize=(10,6))
corr = dataTrain.corr()  
sns.heatmap(corr,annot=True)
plt.show()


# In[9]:


dataTrain.describe()


# In[10]:


dataTrain['price'].plot(kind = 'hist', bins = 5, edgecolor='black')   # 5 bins are used
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()


# In[11]:


dataTrain.describe(include = 'object')


# In[12]:


plt.figure(figsize=(10,6))
sns.regplot(x="bedrooms", y="price", data=dataTrain)


# In[13]:


from scipy import stats
pearson_coef, p_value = stats.pearsonr(dataTrain['bedrooms'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# In[14]:


plt.figure(figsize=(10,6))
sns.regplot(x="sqft_living", y="price", data=dataTrain)


# In[15]:


pearson_coef, p_value = stats.pearsonr(dataTrain['sqft_living'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# In[16]:


plt.figure(figsize=(10,6))
sns.regplot(x="sqft_lot", y="price", data=dataTrain)


# In[17]:


pearson_coef, p_value = stats.pearsonr(dataTrain['sqft_lot'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# In[18]:


plt.figure(figsize=(10,6))
sns.regplot(x="floors", y="price", data=dataTrain)


# In[19]:


pearson_coef, p_value = stats.pearsonr(dataTrain['floors'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[20]:


plt.figure(figsize=(10,6))
sns.regplot(x="condition", y="price", data=dataTrain)


# In[21]:


pearson_coef, p_value = stats.pearsonr(dataTrain['condition'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[22]:


plt.figure(figsize=(10,6))
sns.regplot(x="grade", y="price", data=dataTrain)


# In[23]:


pearson_coef, p_value = stats.pearsonr(dataTrain['grade'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[24]:


plt.figure(figsize=(10,6))
sns.regplot(x="sqft_above", y="price", data=dataTrain)


# In[25]:


pearson_coef, p_value = stats.pearsonr(dataTrain['sqft_above'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[26]:


plt.figure(figsize=(10,6))
sns.regplot(x="sqft_basement", y="price", data=dataTrain)


# In[27]:


pearson_coef, p_value = stats.pearsonr(dataTrain['sqft_basement'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[28]:


plt.figure(figsize=(10,6))
sns.regplot(x="yr_built", y="price", data=dataTrain)


# In[29]:


pearson_coef, p_value = stats.pearsonr(dataTrain['yr_built'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[30]:


plt.figure(figsize=(10,6))
sns.regplot(x="yr_renovated", y="price", data=dataTrain)


# In[31]:


pearson_coef, p_value = stats.pearsonr(dataTrain['yr_renovated'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[32]:


plt.figure(figsize=(10,6))
sns.regplot(x="zipcode", y="price", data=dataTrain)


# In[33]:


pearson_coef, p_value = stats.pearsonr(dataTrain['zipcode'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[34]:


plt.figure(figsize=(10,6))
sns.regplot(x="lat", y="price", data=dataTrain)


# In[35]:


pearson_coef, p_value = stats.pearsonr(dataTrain['lat'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[36]:


plt.figure(figsize=(10,6))
sns.regplot(x="long", y="price", data=dataTrain)


# In[37]:


pearson_coef, p_value = stats.pearsonr(dataTrain['long'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[38]:


plt.figure(figsize=(10,6))
sns.regplot(x="sqft_living15", y="price", data=dataTrain)


# In[39]:


pearson_coef, p_value = stats.pearsonr(dataTrain['sqft_living15'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[40]:


plt.figure(figsize=(10,6))
sns.regplot(x="sqft_lot15", y="price", data=dataTrain)


# In[41]:


pearson_coef, p_value = stats.pearsonr(dataTrain['sqft_lot15'], dataTrain['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[42]:


sns.boxplot(x="bedrooms", y="price", data=dataTrain)


# In[43]:


plt.figure(figsize=(10,6))
sns.boxplot(x="floors", y="price", data=dataTrain)


# In[44]:


plt.figure(figsize=(10,6))
sns.boxplot(x="bathrooms", y="price", data=dataTrain)


# In[45]:


plt.figure(figsize=(12,6))
sns.boxplot(x="sqft_living", y="price", data=dataTrain)


# In[46]:


plt.figure(figsize=(10,6))
sns.boxplot(x="sqft_lot", y="price", data=dataTrain)


# In[47]:


plt.figure(figsize=(10,6))
sns.boxplot(x="waterfront", y="price", data=dataTrain)


# In[48]:


plt.figure(figsize=(10,6))
sns.boxplot(x="view", y="price", data=dataTrain)


# In[49]:


plt.figure(figsize=(10,6))
sns.boxplot(x="condition", y="price", data=dataTrain)


# In[50]:


plt.figure(figsize=(10,6))
sns.boxplot(x="grade", y="price", data=dataTrain)


# In[51]:


plt.figure(figsize=(10,6))
sns.boxplot(x="sqft_above", y="price", data=dataTrain)


# In[52]:


plt.figure(figsize=(10,6))
sns.boxplot(x="sqft_basement", y="price", data=dataTrain)


# In[53]:


plt.figure(figsize=(10,6))
sns.boxplot(x="yr_built", y="price", data=dataTrain)


# In[54]:


plt.figure(figsize=(10,6))
sns.boxplot(x="yr_renovated", y="price", data=dataTrain)


# In[55]:


plt.figure(figsize=(10,6))
sns.boxplot(x="zipcode", y="price", data=dataTrain)


# In[56]:


plt.figure(figsize=(10,6))
sns.boxplot(x="lat", y="price", data=dataTrain)


# In[57]:


plt.figure(figsize=(10,6))
sns.boxplot(x="long", y="price", data=dataTrain)


# In[58]:


plt.figure(figsize=(10,6))
sns.boxplot(x="sqft_living15", y="price", data=dataTrain)


# In[59]:


plt.figure(figsize=(10,6))
sns.boxplot(x="sqft_lot15", y="price", data=dataTrain)


# In[60]:


plt.figure(figsize=(10,6))
sns.boxplot(x="date", y="price", data=dataTrain)


# In[61]:


dataTrain.drop(['sqft_lot','condition','zipcode','sqft_lot15'], axis = 1, inplace = True)


# In[62]:


dataTrain.shape


# In[63]:


# data Transformation
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
dataTrain.date = labelencoder.fit_transform(dataTrain.date)
dataTrain.bedrooms = labelencoder.fit_transform(dataTrain.bedrooms)
dataTrain.bathrooms = labelencoder.fit_transform(dataTrain.bathrooms)
dataTrain.sqft_living = labelencoder.fit_transform(dataTrain.sqft_living)


dataTrain.floors = labelencoder.fit_transform(dataTrain.floors)
dataTrain.waterfront = labelencoder.fit_transform(dataTrain.waterfront)
dataTrain.view = labelencoder.fit_transform(dataTrain.view)

dataTrain.yr_built = labelencoder.fit_transform(dataTrain.yr_built)
dataTrain.yr_renovated = labelencoder.fit_transform(dataTrain.yr_renovated)
dataTrain.long = labelencoder.fit_transform(dataTrain.long)

dataTrain.sqft_above = labelencoder.fit_transform(dataTrain.sqft_above)
dataTrain.sqft_living15 = labelencoder.fit_transform(dataTrain.sqft_living15)
dataTrain.grade = labelencoder.fit_transform(dataTrain.grade)
dataTrain.lat = labelencoder.fit_transform(dataTrain.lat)
dataTrain.sqft_basement = labelencoder.fit_transform(dataTrain.sqft_basement)


# In[64]:


dataTrain.head(10)


# In[65]:


# data Transformation(normalization)
import scipy.stats as stats
dataTrain = stats.zscore(dataTrain)


# In[66]:


dataTrain


# In[67]:


import numpy as np
from sklearn.model_selection import train_test_split
x = dataTrain.drop(['price'], axis=1)
y = dataTrain['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[68]:


dataTrain.shape


# In[69]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[70]:


x_train.head()


# In[71]:


y_train.head()


# In[72]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model_mlr = model.fit(x_train,y_train)


# In[73]:


y_pred1 = model_mlr.predict(x_test)


# In[74]:


print(model_mlr.intercept_)
print(model_mlr.coef_)


# In[75]:


y_test[0:]


# In[76]:


y_pred1[0]


# In[77]:


# MLR Evaluation( Calculating the Mean Square Error for MLR model)
mse1 = mean_squared_error(y_test, y_pred1)
print('The mean square error for Multiple Linear Regression: ', mse1)


# In[78]:


# Calculating the Mean Absolute Error for MLR model
mae1= mean_absolute_error(y_test, y_pred1)
print('The mean absolute error for Multiple Linear Regression: ',mae1)


# In[79]:


#Random Forest Regressor (checking other Models)
# Calling the random forest model and fitting the training data
rf = RandomForestRegressor()
model_rf = rf.fit(x_train,y_train)


# In[80]:


# Prediction of house prices using the testing data
y_pred2 = model_rf.predict(x_test)


# In[81]:


#Random Forest Evaluation
# Calculating the Mean Square Error for Random Forest Model (Lowest MSE value)
mse2 = mean_squared_error(y_test, y_pred2)
print('The mean square error of price and predicted value is: ', mse2)


# In[82]:


# Calculating the Mean Absolute Error for Random Forest Model (Lowest Mean Absolute Error)
mae2= mean_absolute_error(y_test, y_pred2)
print('The mean absolute error of price and predicted value is: ', mae2)


# In[83]:


#LASSO Model 
# Calling the model and fitting the training data
LassoModel = Lasso()
model_lm = LassoModel.fit(x_train,y_train)


# In[84]:


# Price prediction uisng testing data
y_pred3 = model_lm.predict(x_test)


# In[85]:


#LASSO Evaluation  (checking another model)
# Mean Absolute Error for LASSO Model
mae3= mean_absolute_error(y_test, y_pred3)
print('The mean absolute error of price and predicted value is: ', mae3)


# In[86]:


# Mean Squared Error for the LASSO Model
mse3 = mean_squared_error(y_test, y_pred3)
print('The mean square error of price and predicted value is: ', mse3)


# In[87]:


scores = [('MLR', mae1),
          ('Random Forest', mae2),
          ('LASSO', mae3)
         ]         


# In[88]:


mae = pd.DataFrame(data = scores, columns=['Model', 'MAE Score'])
mae


# In[89]:


mae.sort_values(by=(['MAE Score']), ascending=False, inplace=True)

f, axe = plt.subplots(1,1, figsize=(10,7))
sns.barplot(x = mae['Model'], y=mae['MAE Score'], ax = axe)
axe.set_xlabel('Model', size=20)
axe.set_ylabel('Mean Absolute Error', size=20)

plt.show()


# In[ ]:




