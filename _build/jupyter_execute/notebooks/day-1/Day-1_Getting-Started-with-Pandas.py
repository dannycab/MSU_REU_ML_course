#!/usr/bin/env python
# coding: utf-8

# # Getting Started with Pandas

# In this notebook we will be:
# 
# 1. Reviewing the basics of Pandas
# 2. Reviewing Pandas Dataframes
#  
# ## Notebook instructions
# 
# Recall that to make notebook cells that have Python code in them do something, hold down the 'shift' key and then press the 'enter' key (you'll have to do this to get the YouTube videos to run).  To edit a cell (to add answers, for example) you double-click on the cell, add your text, and then enter it by holding down 'shift' and pressing 'enter'

# ### Imports for this Notebook
# 
# One of the downsides of notebooks is knowing when things got imported and what modules were important. Trying to get into the habit of including all of the important imports at the top of the notebook is a good way to ensure that you don't run into issues later on in the notebook. When you restart the notebook, you can run that first cell to get the imports right.

# In[1]:


from IPython.display import HTML
from IPython.display import YouTubeVideo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ----
# 
# ## 1. The basics of Pandas
# 
# Pandas is a fundamental tool for doing Data Science work in Python. While we can do this work in straight Python, Pandas makes much of that work easier. We cannot do justice to all of Pandas, as it is a big package, but here we'll remind ourselves of some of the basics. As you do more data science work, you'll pick more and more of Pandas as you go along. 

# ### 1.1 Pandas Series
# 
# The basic Pandas data structure is called a *Series*. It is a sequence, not unlike a numpy array, but with an associated set of labels for each value called the *index*. If you don't provide the index labels, Pandas will use the regular 0-based index as the label. Again: if you don't provide index labels, it will use the numeric index as the index label. That will be important later.
# 
# You can make a Series using either a python *dict*, with keys as the indices, or by separately providing the values and indices. You can also update the index labels or reset the labels to the default. Note however that the `reset_index` method does *not* change the Series but returns, not a Series but a DataFrame, where the original index is preserved as a new column.
# 
# &#9989; **Review the following example and make sure you understand everything that is happening. Discuss with your group mates.**

# In[2]:


# assumes you have imported Pandas as pd in the cell at the top of this page
series_index = pd.Series([1,2,3,4], index=['one', 'two', 'three', 'four'])
print("\nSeries with indicies")
print("Type:", type(series_index))
print(series_index)

series_noindex = pd.Series([5,6,7,8])
print("\nSeries with default indices")
print("Type:", type(series_noindex))
print(series_noindex)

my_dictionary = { 'nine': 9, 'ten':10, 'eleven':11, 'twelve':12 }
series_dictionary = pd.Series(my_dictionary)
print("\nSeries from a dictionary")
print(series_dictionary)


# ### 1.2 Manipulating Series
# 
# Once you have a Pandas Series object, You can access the values in a number of ways:
# * using the label in [ ], much as you would in a dictionary
# * using data member "dot" (`.`) access, if the label name would constitute a valid Python variable name (can't start with a digit for example)
# * using numpy array indexing
# 
# Without a label (using default indices) you are restricted to using only the last approach.
# 
# &#9989; **Review the following mechanisms for accessing data in a Pandas series based on the format and structure of the Series object**.

# In[3]:


#using label
print(series_index["three"])

#using data member access
print(series_index.three)

#using array index, 0-based
print(series_index[2])

# no labels
print(series_noindex[2])
# series_noindex.2   # can't, 2 isn't a valid Python variable name


# Once you have a series object, **you can assign/change the values to any of the locations that you can access**. Like so:

# In[4]:


print("Before:")
print(series_dictionary)
print("---")

series_dictionary["eleven"] = 111
series_dictionary.twelve = 122

print("After:")
print(series_dictionary)


# ### 1.3 Numpy like operations
# Finally, you can do many of the things you can do with NumPy arrays, such as indexing NumPy arrays, with a Pandas Series object as well.
# 
# &#9989; **Review the following examples to convince yourself how you can use NumPy-style operations to access Series data in Pandas. Discuss with your group mates.**
# 
# Look at how you can work with ranges of the series elements. The labels are ordered and so the following works:

# In[5]:


print(series_index["two":])


# You can also apply **Boolean masks to a Series**:

# In[6]:


print(series_dictionary[series_dictionary <= 10])


# And you can **perform operations** which return a new series (but don't modify the existing one):

# In[7]:


print(series_dictionary * 2)
print(series_dictionary.mean() )


# There are **many operations** you can perform on a Pandas Series object (over 200 last we checked!). You'll pick up more as you continue to become a Pandas expert.

# ----
# 
# ## 2. The Pandas Dataframe
# 
# A Pandas DataFrame is a 2 dimensional data structure. The easiest way to think of a DataFrame is as a group of Series objects where each Series represents a column in the 2D structure. As with Series you can make them a number of ways but the standard way is to use a dictionary where the keys are the column *headers* and the values are a list of values under that header.
# 
# <div align="left"><img src="https://i.ibb.co/C13ybjZ/df.png" alt="df" border="0" width=400></div>
# 
# It is always important to know the **types** in each column as that can affect the kinds of operations you can perform on a column. Listing the `.dtypes` provides such a list. A type of `object` is likely (though not necessarily) a string.
# 
# An index for the rows is provided by default using 0-based array indexing. The use of `[]` label indexing returns a `Series` which is a column with that heading name. The index of the entire DataFrame is used for the returned Series.
#     
# &#9989; **Run and review the following code: Discuss the results with your group mates.**

# In[8]:


patient_dict = {"name":["james","jim","joan","jill"],
                 "age":[10, 20, 30, 40],
                 "weight":[150, 140, 130, 120],
                 "complaint": ["sprain", "cut", "headache", "break"]}
patient_df = pd.DataFrame(patient_dict)

print(type(patient_df))
print(patient_df)

print("\n Column types")
print(patient_df.dtypes)

print("\n age column")
age = patient_df["age"]
print(age)
print(type(age))


# ### 2.1 Data Frame indexing
# 
# As we noted above, the index for a DataFrame by default is a 0-based array index. However, we can choose to use a different column as an index for the DataFrame. The `.set_index()` method allows us to set the index.
# 
# An **irritating** issue is whether the Series being worked on changes. Typically it does not but most methods provide a `inplace = True` setting to apply the changes to the existing element. Otherwise you must assign the result to preserve it.
# 
# If you do not do so, the column being used as an index is no longer available using standard indexing. If you provide `drop = False` in set_index, the column is preserved in the DataFrame as well as in the index. If you `.reset_index()` returns to 0-based indexing
# 
# To access a row, you can use either `.loc` or `.iloc`
# * `.loc` uses the row index **label** to access the row (or the 0-based index of none is provided). It returns a Series
# * `.iloc` uses the 0-based index regardless of whether a label exists. It too returns a Series
# 
# Indicies and headers are preserved in the Series indexed from a DataFrame
# 
# &#9989; **Take a look at the following and make sure you can follow how the dataframe is being manipulated. Discuss with your group mates.**

# In[9]:


patient_df.set_index("age", inplace=True)
print("\nAge is now the index")
print(patient_df)

# reset to 0-based
patient_df.reset_index(inplace=True)
print("\nBack to 0-based indexing")
print(patient_df)

# keep age as a column
new_df = patient_df.set_index("age", drop=False)
print("\nDon't change the original")
print(patient_df)
print("\nIndex by age, keep the age column")
print(new_df)


# ### 2.2 Try it yourself
# 
# &#9989; **Try doing the following**:
# * Make a DataFrame to store student grades. The column headers should be:
#     * Name
#     * ID
#     * Total Percent (out of 100)
#     * Final Grade Point (on the 0.0 - 4.0 scale)
# 
# Make up some names and values to fill your Dataframe. Include **at least 8 students**.
# 
# Then: 
# * Set the index to be the ID
# * Print every student in the dataframe who got a 3.0 or greater

# In[10]:


# Put your code here


# ---
# ## Follow-up Questions
# 
# 1. Is there anything involving using Pandas that you're curious about or are there any specific issues you've run into in the past with Pandas that you couldn't find a solution for?

# In[ ]:




