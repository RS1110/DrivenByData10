
# coding: utf-8

# ## Importing Dependencies
# 

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sns


# ## Loading Data

# In[2]:

train=pd.read_csv('/resources/data/Data/train.csv')
test=pd.read_csv('/resources/data/Data/test.csv')
train.head(10)


# In[3]:

import matplotlib.pyplot as plt
corr=train.corr()
get_ipython().magic(u'matplotlib inline')
plt.subplots(figsize=(16,12))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# In[4]:

##From the above heatmap, we can clearly see that there's a correlation between the two factors : due_mortgage and value. 


# In[5]:

r=train["DUE_MORTGAGE"]/train["VALUE"]
r.head(5)


# In[6]:

ratio=r.mean()
ratio


# In[7]:

##Ratio will be used to find out missing values of due_mortgage and value (only if one of them is missing).


# ## Combining Test and Train Datasets

# In[8]:

#combining test and train datasets

df= pd.concat([train, test], axis=0, join='outer')


# ## Preprocessing - Handling Missing Values

# In[9]:


df.isnull().sum()


# In[10]:


#VALUE and DUE_MORTGAGE are in a ratio
#New feature was added in df_new 

df_new=(df.loc[:,['DUE_MORTGAGE', 'VALUE']]).dropna()
df_new['ratio'] = df_new['VALUE'] / df_new['DUE_MORTGAGE']

VbyD = df_new.loc[:,'ratio'].mean()

print(VbyD)


# In[11]:

#When both the features of Due_mortgage and value are Nan, 
#individual means of the columns of Due_mortgage and value are calcuated
#missing values of both the columns were calculated using VbyD

df.loc[df['DUE_MORTGAGE'].isnull() & df['VALUE'].isnull(), 'DUE_MORTGAGE'] = df['DUE_MORTGAGE'].mean()
df.loc[df['DUE_MORTGAGE'].isnull() & df['VALUE'].isnull(), 'VALUE'] = df['VALUE'].mean()

df['DUE_MORTGAGE'].fillna(df['VALUE']/VbyD, inplace=True)
df['VALUE'].fillna(df['DUE_MORTGAGE']*VbyD, inplace=True)


# In[12]:

df.isnull().sum()


# In[13]:

df['REASON'].fillna(df['REASON'].mode()[0], inplace=True)
df['OCC'].fillna(df['OCC'].mode()[0], inplace=True)
df['DCL'].fillna(df['DCL'].mode()[0], inplace=True)
df['CLT'].fillna(df['CLT'].mean(), inplace=True)
df['TJOB'].fillna(df['TJOB'].mean(), inplace=True)  #doubtful
df['CL_COUNT'].fillna(df['CL_COUNT'].mean(), inplace=True)
df['RATIO'].fillna(-1, inplace=True)
# -1 is dummy value. RF model will handle these missing values of ratio accordingly.
df.isnull().sum()


# ## Feature Engineering

# In[14]:


df['income']=df['AMOUNT']/df['RATIO'] #already given

s=df[df['income']>0].groupby(['OCC']).mean()['income']

df.loc[(df['income']<0) & (df['OCC']==0.0), 'income'] = s[0.0]

df.loc[(df['income']<0) & (df['OCC']==1.0), 'income'] = s[1.0]

df.loc[(df['income']<0) & (df['OCC']==2.0), 'income'] = s[2.0]

df.loc[(df['income']<0) & (df['OCC']==3.0), 'income'] = s[3.0]

df.loc[(df['income']<0) & (df['OCC']==4.0), 'income'] = s[4.0]

df.loc[(df['income']<0) & (df['OCC']==5.0), 'income'] = s[5.0]

df['RATIO']=df['AMOUNT']/df['income']

#New feature - Due_mortgage/value

df['ratio_1']=df['DUE_MORTGAGE']/df['VALUE']


# ## Modelling

# ## Splitting train and validation set

# In[15]:


features = ['AMOUNT', 'CLT', 'CL_COUNT', 'CONVICTED', 'DCL',
       'DUE_MORTGAGE', 'OCC', 'RATIO', 'REASON', 'TJOB',
       'VALUE', 'VAR_1', 'VAR_2', 'VAR_3', 'income', 'ratio_1']

X = df[df['TEST_ID'].isnull()].loc[:,features]
Y = df[df['TEST_ID'].isnull()].loc[:,'DEFAULTER']

X_test_resplit = df[df['TEST_ID'].notnull()].loc[:,features]

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split( X, Y,
                                        test_size=0.33, random_state=12, stratify=Y)pred_test = model.predict_proba(X_test_sub)[:,1]


# ## RF Model

# In[16]:

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

model = RandomForestClassifier(n_estimators=300,max_leaf_nodes = 15,max_depth=4,
                              min_samples_leaf=11, min_samples_split= 20,
                              n_jobs=-1, random_state =7)

model.fit(X_train,Y_train)


# In[ ]:

#Prediction
s=model.predict_proba(X_valid)[:,1]
s_class = model.predict(X_valid)
roc_auc_score(s_class, Y_valid)


# In[ ]:

#Prediciting probability of defaulter value = 1 from test set.
predict_test = model.predict_proba(X_test_resplit)[:,1]
predict_test


# In[ ]:

test['DEFAULTER'] = predict_test
test.loc[:,['LOAN_ID', 'DEFAULTER']].to_csv('/resources/data/Data/final.csv')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



