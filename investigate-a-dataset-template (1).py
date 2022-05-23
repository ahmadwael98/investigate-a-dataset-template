#!/usr/bin/env python
# coding: utf-8

# # > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Before submitting your project, it will be a good idea to go back through your report and remove these sections to make the presentation of your work as tidy as possible. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate a Dataset (noshowappointments.csv)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# # <a id='intro'></a>
# ## Introduction
# >This dataset collects information
# from 100k medical appointments in
# Brazil and is focused on the question
# of whether or not patients show up
# for their appointment. A number of
# characteristics about the patient are
# included in each row.
# ● ‘ScheduledDay’ tells us on
# what day the patient set up their
# appointment.
# ● ‘Neighborhood’ indicates the
# location of the hospital.
# ● ‘Scholarship’ indicates
# whether or not the patient is
# enrolled in Brasilian welfare
# program Bolsa Família.
# ● Be careful about the encoding
# of the last column: it says ‘No’ if
# the patient showed up to their
# appointment, and ‘Yes’ if they
# did not show up

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
sns.set(rc={'figure.figsize':[8,8]},font_scale=1.2)


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you document your steps carefully and justify your cleaning decisions.
# 
# ### General Properties

# In[2]:


df=pd.read_csv('noshowappointments.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.corr()


# In[6]:


df.info()


# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.
# 
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
# 
# ### Data Cleaning (Replace this with more specific notes!)

# In[7]:


# we will not need the patient id or the appointment id
df.drop(['PatientId','AppointmentID'],axis=1,inplace=True)
df.head()


# In[8]:


df[['ScheduledDay']].head()


# In[9]:


#to make sure there is no nana data
df['ScheduledDay'].isna().mean()


# In[10]:


#to make sure there is no nana data
df['AppointmentDay'].isna().mean()


# In[11]:


#to make sure there is no nana data
df['Age'].isna().mean()


# In[12]:


#extracting the day so we can use it later
df['ScheduledDay']=pd.to_datetime(df['ScheduledDay'])
df['scheduledday']=df['ScheduledDay'].dt.day_name()


# In[13]:


#extracting the day so we can use it later
df['AppointmentDay']=pd.to_datetime(df['AppointmentDay'])
df['appointmentday']=df['AppointmentDay'].dt.day_name()


# In[14]:


#correcting hipertemsion to hypertension
df.rename(columns={'Hipertension' : 'Hypertension'},inplace=True)


# In[15]:


#renaming the column
df.rename(columns={'No-show' : 'no_show'},inplace=True)


# In[16]:


#observing the age column
df['Age'].unique()


# In[17]:


df[df['Age']<=0].shape


# In[18]:


#cleaning the Age column
def cleaning(row):
    if row['Age']<=0:
        return 0
    else:
        return row['Age']


# In[19]:


df['Age']=df.apply(cleaning, axis=1)


# In[20]:


df[df['Age']<0].shape


# # the number of people who showed is 4 times those who showed

# In[21]:


df[df['no_show']=='No'].shape


# In[22]:


df[df['no_show']=='Yes'].shape


# In[23]:


df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. Compute statistics and create visualizations with the goal of addressing the research questions that you posed in the Introduction section. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables.
# 
# ### General  histogram for all the data

# In[24]:


#general look about the data
df.hist(figsize=(17,15))


# # -most people didn't drink
# # -most people didn't have diabetes
# # -most people didn't recieve a message
# # -most people weren't handcap

# In[25]:


df.head()


# In[26]:


def countplot(x_axis,hue):
    sns.countplot(x=x_axis,data=df,hue=hue)
def displot(x_axis,hue):
    sns.displot(x=x_axis,data=df,hue=hue)

    


# In[27]:


#assigning yes to no_show and no to show
show = df.no_show == 'No'
no_show = df.no_show == 'Yes'
df[show].count()


# In[28]:


#checking the percentage of the males  and females according to no_show data
countplot('no_show','Gender')


# ## Most patients are females and they are the ones who showed more than the males the percentage of the females who showed are higher than males.

# In[29]:


#there is no reasonable data that we can extract from the appointment day as all the data are similar
countplot('no_show','appointmentday')


# In[30]:


df[no_show].count()


# In[31]:


#checking the relation between Neighbourhood and no_show
plt.figure(figsize=[14,9])
df.Neighbourhood[show].value_counts().plot(alpha = 1 , label='show',kind='bar' )
df.Neighbourhood[no_show].value_counts().plot(alpha = 1 , label='no_show',kind='bar',color='yellow')
plt.legend()
plt.title('relation between Neighbourhood and no_show')
plt.xlabel('Neighbourhood')
plt.ylabel('Number of patients')


# In[32]:


#Comparing patients attendance according to their Age
plt.figure(figsize=[14,9])
df.Age[show].hist(alpha = 1 , label='show' )
df.Age[no_show].hist(alpha = 1 , label='no_show')
plt.legend()
plt.title('Comparing patients attendance according to their Age')
plt.xlabel('Age')
plt.ylabel('Number of patients')


# In[33]:


#checking the relation between the sms recieved and the no_show data
countplot('SMS_received','no_show')


# In[34]:


#checking the relation between the handcap and the no_show data
countplot('no_show','Handcap')


# In[35]:


#checking the relation between the alcoholism and the no_show data
displot('no_show','Alcoholism')


# In[36]:


#checking the relation between the diabetes and the no_show data

countplot('no_show','Diabetes')


# In[37]:


#checking the relation between the hypertension and the no_show data

countplot('no_show','Hypertension')


# <a id='conclusions'></a>
# ## Conclusions
# 
# Age is a huge factor 0-10 , 45-55 has the highest rate of attendance while 22-37 has the lowest rate of attendance.
# 
# Generally most people who booked  are in range of 0-10.
# 
# Most patients are females and they are the ones who showed more than the males the percentage of the females who showed are higher than males.
# 
# Diabetes, Hypertension, Handicap and Alcoholism are not effective in the dataset.
# 
# sms data are messed up as the people who didn’t get the sms showed more than the people who got it but we can conclude from the data shown that the sms aren’t important so the clinics can stop sending the sms to decrease the expenses .
# 
# from the dataset we can conclude that the most affecting factor is the neighbourhood.
# 
# the day didnt affect the noshow percentage
# 

# ## limitations 
# most columns werent important
# 
# there are columns that were useless that we had to drop
# 
# there were some columns that werent making any sense (Sms)
# 
# 
# 
# 

# In[38]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




