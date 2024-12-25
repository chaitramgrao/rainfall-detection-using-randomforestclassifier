#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[36]:


data = pd.read_csv(r'C:\Users\CHAITRA M G\Downloads\Weather_Data.csv')


# In[37]:


# Check for missing values
print(data.isnull().sum())


# In[38]:


data.head()


# In[39]:


data.tail()


# In[40]:


data.dropna(inplace=True)


# In[41]:


data.shape


# In[42]:


data.columns


# In[44]:


data.count()


# In[45]:


data.Weather.unique()


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)


# In[47]:


data.dtypes


# In[48]:


data.nunique()


# In[43]:


features = data[['Temp_C', 'Dew Point Temp_C', 'Wind Speed_km/h', 'Press_kPa']]
target = data['Weather']


# In[49]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[50]:


model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[51]:


model.fit(X_train, y_train)


# In[52]:


y_pred = model.predict(X_test)


# In[53]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[54]:


new_data = [[30, 70, 12, 1012]]  # [Temp, Dew point temp, WindSpeed, Pressure]
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Prediction :", prediction[0])


# In[ ]:




