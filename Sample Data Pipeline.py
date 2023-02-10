#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pickle
import pandas as pd


# In[31]:


# Load in Eoin's saved sample data and models and scaler

df = pd.read_csv('sample_data.csv')

# scaler for transforming data
file = open("scaler.pkl",'rb')
scaler = pickle.load(file)
file.close()

# ML model
file = open("mlp.pkl",'rb')
mlp = pickle.load(file)
file.close()

# Screening model
file = open("screening_model.pkl",'rb')
screening_model = pickle.load(file)
file.close()


# In[33]:


# Just indexing column not needed
del df['Unnamed: 0']

# Target growth rates to predict
target = df.growth_rate.values
del df['growth_rate']


# In[34]:


# Transform data for model input
scaled_data = scaler.transform(df)


# In[35]:


df.head()


# ## It's probably easy to just define a function for the whole hybrid model

# In[49]:


def hybrid_model(screening_model, mlp, most, x):
    """
    Input: Screening model, Machine learning model MLP, MoSt model, queries x
    Return: Prediction for datapoint x
    """
    
    final_predictions = list()
    
    screening_preds = screening_model.predict(scaled_data)
    mlp_preds = mlp.predict(x)
    mlp_preds[mlp_preds < 0] = 0
    most_preds = most.predict(x)
    
    for i in range(len(screening_preds)):
        model_to_use = screening_preds[i]
        
        if model_to_use == 1:
            final_predictions.append(mlp_preds[i])
            
        elif model_to_use == 0:
            final_predictions.append(most_preds[i])
            
    return final_predictions


# In[50]:


hybrid_predictions = hybrid_model(screening_model, mlp, most, scaled_data)


# In[ ]:




