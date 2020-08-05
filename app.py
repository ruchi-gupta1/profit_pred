#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data=pd.read_csv('50_Startups.csv')
data=data.drop('State', axis=1)
data.head()


# In[3]:


x=data.iloc[:,0:-1]
y=data.iloc[:,-1]


# In[4]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)


# In[5]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[6]:


model.fit(xtrain, ytrain)


# In[7]:


ypred=model.predict(xtest)


# In[8]:


from sklearn.metrics import r2_score
r2_score(ytest, ypred)


# In[ ]:


from flask import Flask,render_template,request
app=Flask(__name__)
@app.route("/")
def ruchi():
    return render_template("ruchi.html")
@app.route("/details", methods=['POST','GET'])
def ayu():
    if(request.method=="POST"):
        a=float(request.form["v1"])
        b=float(request.form["v2"])
        c=float(request.form["v3"])
        result=model.predict([[a,b,c]])
        return render_template("ruchi.html",taani=result)
app.run()


# In[1]:





# In[ ]:




