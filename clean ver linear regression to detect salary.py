#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("Salary.csv")


# In[4]:


df.info()


# In[14]:


# checking outliers
plt.subplot(1,2,1)
yearsb= plt
yearsb.boxplot("YearsExperience",data= df)
yearsb.title("Boxplot of Years")
yearsb.ylabel("Years")
yearsb.xlabel("boxplot")
yearsb.xticks([1],["Year"])
yearsb.tight_layout()

plt.subplot(1,2,2)
salaryb= plt
salaryb.boxplot("Salary", data= df)
salaryb.title("Boxplot of Salary")
salaryb.ylabel("Salary")
salaryb.xlabel("boxplot")
salaryb.xticks([1],["Salary"])
salaryb.tight_layout()


# In[25]:


#cheching correlation
sns.heatmap(df.corr(),annot=True)


# In[19]:


#defining x, y
x= df["YearsExperience"].values.reshape(-1,1)
y= df["Salary"].values.reshape(-1,1)
print(x.shape, y.shape)


# In[24]:


# data splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,random_state=42, train_size=0.8)
print( x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[26]:


# linear regression
from sklearn.linear_model import LinearRegression
basemodel= LinearRegression()
basemodel.fit(x_train,y_train)


# In[35]:


# model evaluation
from sklearn.metrics import r2_score
r2_value = r2_score(y_train,basemodel.predict(x_train))
intercept= basemodel.intercept_
coefficient= basemodel.coef_

print(f"r2 value is: ", r2_value)
print(f"the linear regression equation for this model is : ", "y= ", intercept, "+", coefficient,"x")


# In[39]:


# prediction for y test
y_pred = basemodel.predict (x_test)

#checking evaluation
r2_value1= r2_score(y_test,y_pred)
print(r2_value1)


# In[47]:


#visualize findings
plt.subplot (1,2,1)
viz_train= plt
viz_train.scatter(x_train,y_train)
viz_train.plot(x_train,basemodel.predict(x_train), color= "black")
viz_train.title("Comparison of y train vs y predict")
plt.tight_layout()

plt.subplot (1,2,2)
viz_test= plt
viz_test.scatter(x_test,y_test)
viz_test.plot(x_test,basemodel.predict(x_test),color= "black")
viz_test.title("Comparison of y test vs y predict")
plt.tight_layout()


# In[49]:


basemodel.predict([[2.5]])


# In[55]:


# user input to predict
inputation= input("how many years of experience do you have? ")
print("your salary could be as high as: ", basemodel.predict([[inputation]]))


# In[ ]:




