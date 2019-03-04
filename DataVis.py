
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('./Twitter_sentiment_DJIA30/twitter_data_NKE.csv')


# In[3]:


from events import events_from_data


# In[7]:


event_idxs, filtered_idxs = events_from_data(data, L=10)


# In[7]:


#data.loc[filtered_idxs]


# In[8]:


#data.loc[event_idxs]

