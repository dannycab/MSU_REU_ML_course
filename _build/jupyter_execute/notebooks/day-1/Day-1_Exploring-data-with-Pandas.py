#!/usr/bin/env python
# coding: utf-8

# # Exploring data with Pandas
# 
# <img src="https://miro.medium.com/max/819/1*Dss7A8Z-M4x8LD9ccgw7pQ.png" width="500px">

# Today, we will:
# 
# 1. Make sure that everyone remembers how to do the basics with `pandas`.
# 2. Do some data analysis with existing data sets.
# 3. Make some visualizations of the data.
# 
# ## Notebook instructions
# 
# Work through the notebook making sure to write all necessary code and answer any questions.
# 
# ### Outline:
# 
# 1. [Review of `pandas`](#review)
# 2. [Loading and exploring a dataset](#loading)
# 3. [Visualizing your data](#visualizing)

# ### Useful imports (make sure to execute this cell!)
# Let's get a few of our imports out of the way. If you find others you need to add, consider coming back and putting them here.

# In[1]:


# necessary imports for this notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from pandas.plotting import scatter_matrix


# ----
# <a id="review"></a>
# ## 1. Review of `pandas`
# 
# Let's take a moment to highlight some key concepts. **Discuss with your group mates** the following prompts and write down a brief definition of each of these concepts.
# 
# **If you don't feel like you have good working definitions yet, try doing a quick internet search to see if you can find a definition that makes sense to you.**
# 
# &#9989; **Question 1:** What are the features of a Pandas Series?

# <font size=+3>&#9998;</font> Do This -  Write your discussion notes here.

# &#9989; **Question 2:** What are the differences between a DataFrame and a Series?

# <font size=+3>&#9998;</font> Do This -  Write your discussion notes here.

# ---
# <a id="loading"></a>
# ## 2. Loading and exploring a dataset
# 
# The goal is typically to read some sort of preexisting data **into** a DataFrame so we can work with it. 
# 
# Pandas is pretty flexible about reading in data and can read in a [variety of formats](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html). However, it sometimes needs a little help. Let's start with a "toy" dataset, the <a href="https://en.wikipedia.org/wiki/Iris_flower_data_set"> Iris Data set </a>. 
# 
# A toy dataset is often one that has a particularly nice set of features or is of a manageable size such that it can be used as a good dataset for learning new data analysis techiques or testing out code. This allows one to focus on the code and data science methods without getting too caught up in wrangling the data. However, **data wrangling** is an authentic part of doing any sort of meaningful data analysis as data more often messier than not. 
# 
# Although you will be working with a toy dataset today, you may still have to do a bit of wrangling along the way.
# 
# ### 2.1 Getting used to looking for useful information on the internet
# 
# Another authentic part of working as a computational professional is getting comfortable with searching for help on the internet when you get stuck or run into something that is unfamiliar to you. The Python data analysis/computional modeling world can be complicated and is ever-evolving. There is also a large number of publicly available toolsets with varying levels of complexity and nuance. Through practice and experience, we become better computational folks by learning how to search for better, more efficient, and clearer ways to do things. 
# 
# #### The Iris data
# 
# The iris data set is pretty straight forward (review the wikipedia link above if you haven't yet), especially if someone has done all of the data wrangling and cleaning for you. To make it a little more interesting we provide it in a raw form that you might encounter with other data.
# 
# **&#9989; Do This:**  To get started, **you'll need to download the following two files**:
# 
# `https://raw.githubusercontent.com/dannycab/MSU_REU_ML_course/main/notebooks/day-1/iris.data`
# 
# `https://raw.githubusercontent.com/dannycab/MSU_REU_ML_course/main/notebooks/day-1/iris.names`
# 
# Once you've done so, you should have access to the following : `iris.data` and `iris.names`. Open them both and discuss what you see. This is a good opportunity to use you favorite text editor or use something new. Feel free to ask your group members or instructors what they prefer to use.

# &#9989; **Question 3**: Describe the data and the information in the names file. What characteristics are provided there? Perhaps the iris dataset link above will help.

# <font size=+3>&#9998;</font> Do This - Erase the contents of this cell and replace it with your answer to the above question!  (double-click on this text to edit this cell, and hit shift+enter to save the text). **Practice using appropriate Markdown formating to make your answer's easy to read.**

# ###  2.2 Reading in a file
# 
# Pandas supports a number of file formats, but one of the most common is the one provided by the `read_csv` function. Typically, the phrase **csv** stands for "comma separated values", a text file format that excel and other spread sheets use. In a typical csv file, each line represents a line in a spreadsheet and each cell value is separated from the next by a comma.
# 
# However, you can use other separators as well. Look at the [documentation for `read_csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).

# **&#9989; Do This:** Read the ```iris.data``` file into your notebook **with the appropriate column headers**. Display the DataFrame to make sure it looks reasonable.

# In[2]:


# replace this cell with code to read in the iris.data file with column headers. Use appropriate headers!


# ### 2.3 Describing your data
# 
# The `.describe()` methods tells you about the data. If the data is too big to see (height and width) you can use the two options below to show more of the data. You can change the values to suit your needs. 

# In[3]:


# expand what is shown (rows and cols) by pandas, any time you change default options -- be thoughtful about your choices!
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# **&#9989; Do This:** Run the `.describe()` method on the data you read in. If you think about how you might classify the species contained in the dataset based on the four available features, can you discern anything helpful? Put your answer in the cell below.

# In[4]:


# Put your code here


# <font size=+3>&#9998;</font> Do This - Erase the contents of this cell and replace it with your answer. Did you find any useful features? What makes those features useful in classfying the species of the irises?

# ### 2.4 Grouping your data
# You can perform operations to group elements using the `.groupby()` method. The result of the use of `.groupby` is a `GroupBy` object. If you were to print it you see only the class instance name as nothing is computed until an operation is performed on the groups, much like other iterators. However, you can use `.describe()` on the result of a `.groupby()` call. If you haven't used this before, you may need to consult the [documentation for `.groupby()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) 

# **&#9989; Do This:** Make a GroupBy object using the iris DataFrame and group by the class/species of the data. Then run `.describe()` on the group result. Now take a look at the data from the group. Thinking again about classification, do you see any data features you could use to classify the data when you look at it this way?

# In[5]:


# Put your code here


# <font size=+3>&#9998;</font> Do this - Erase this and put your answer here. Did you find any useful features? What makes those features useful in classfying the species of the irises? How did using `.groupby()` help (or not) in finding useful features?

# -----
# <a id="visualizing"></a>
# ## 3. Visualizing your data
# 
# We are often limited in understanding our data because it is complex, has many features, or is quite large. In these situations, visualizations (plots, charts, and graphs) can help represent our data in ways that help us gain greater insight into the trends, features, and classes we want to understand.

# ### 3.1 Making a scatter matrix
# 
# Rather than just looking at the raw values or exploring basic statistical values for our data, it would be better if we were to visualize the differences. Since the number of features is small, we can use a plotting feature of Pandas called `scatter_matrix()`. Did you notice that we imported `scatter_matrix` at the start of this notebook?
# 
# **&#9989; Do This:** Try calling the `scatter_matrix` function on your dataframe. Use a `marker` that you prefer. Look up the documentation for `scatter_matrix` on the Pandas website for more information! You may need to make the figure bigger to get a good look at it.
# 
# **Note**: There is a similar sort of plot that can be made with the [seaborn](https://seaborn.pydata.org/index.html) package, which you may have seen before. If so, do you remember what it is?

# In[6]:


# your code here


# &#9989; **Question 4**: Does this visualization help you to determine the features that might be useful for classification? Why or why not?

# <font size=+3>&#9998;</font> Do This - Erase the contents of this cell and replace it with your answer to the above question!  (double-click on this text to edit this cell, and hit shift+enter to save the text). **Practice using appropriate Markdown formating to make your answers easy to read.**

# ### 3.2 Color coding your data
# 
# The default scatter matrix probably isn't as helpful as we might hope as it's impossible to tell which points on the plot represent our classes/sepcies.
# 
# We could really use a separate color indication for each dot so we can tell the species apart. Essentially, we need to create an array such that, for each observation in the DataFrame, we can assign a color value to associate it with the appropriate species. It could be just an integer, or it could be one of the standard colors such as "red".
# 
# **&#9989; Do This:** Create a new list, array, or Pandas Series object that maps each species to a particular color. Then recreate the scatter matrix using the `c` argument to give the points in the plot different colors.

# In[7]:


# your code here


# Hope you got something that's a little more useful! OK, how do we read this now?
# * The diagonal shows the distribution of the four  numeric variables of our example data.
# * In the other cells of the plot matrix, we have the scatterplots (i.e. correlation plot) of each variable combination of our dataframe. 

# &#9989; **Question 5**: Are you better able to discern features that might be useful in classfying irises? What can you say about the features of each iris species? Can you separate one species easily? If so, using which feature(s)?

# <font size=+3>&#9998;</font> Do This - Erase the contents of this cell and replace it with your answer to the above question!  (double-click on this text to edit this cell, and hit shift+enter to save the text). **Practice using appropriate Markdown formating to make your answers easy to read.**

# ### 3.3 Separating species of irises

# Now we will use the feature(s) you found above to try to separate iris species. In future parts of the course, we may explore how to do this using models, but for now, you will try to do this by slicing the DataFrame.

# **&#9989; Do This:** One of these species is obviously easier to separate than the others. Can you use a Boolean mask on the dataframe and isolate that one species using the features and *not the species label*? Try to do so below and confirm that you were successful.

# In[8]:


# Put your code here


# **&#9989; Do This:** Can you write a sequence of Boolean masks that separate the other two as well? Are these as effective as the first? Can you think of anything you might do to improve the separation?

# In[9]:


# your code here


# <font size=+3>&#9998;</font> Do This - Erase the contents of this cell and replace it with your answer. How effective your Boolean mask separation of the 3 species? Any ideas for improvements?

# ---
# ## 4. Physics Data Sets
# 
# We used the famous Iris data set to do this activity. That is because it is well known and clearly documented. Most data sets you will work with will not have the same level of documentation. Let's try to find some physics data sets that you can read in and plot.
# 
# Places to look for data: 
# 
# * [Kaggle](https://www.kaggle.com): Kaggle is a well-known platform for data science and machine learning. It offers a vast collection of datasets contributed by the community. You can search for datasets based on various categories, such as image data, text data, time series data, etc.
# * [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php): The UCI Machine Learning Repository is a popular resource for machine learning datasets. It provides a diverse collection of datasets from various domains, including classification, regression, clustering, and more.
# * [Google Dataset Search](https://datasetsearch.research.google.com): Google Dataset Search allows you to search for datasets across different domains. It aggregates datasets from various sources, including academic institutions, data repositories, and other websites.
# * [Data.gov](https://www.data.gov): Data.gov is the U.S. government's open data portal. It provides access to a wide range of datasets covering various topics, such as health, climate, transportation, and more. It's a valuable resource for finding government-related datasets.
# * [Microsoft Research Open Data](https://msropendata.com): Microsoft Research Open Data is a platform that provides access to diverse datasets collected or curated by Microsoft researchers. It includes datasets from domains like computer vision, natural language processing, and healthcare.
# * [AWS Open Data Registry](https://registry.opendata.aws): AWS Open Data Registry is a collection of publicly available datasets provided by Amazon Web Services (AWS). It hosts a variety of datasets, including satellite imagery, genomics data, and more.
# * [OpenML](https://www.openml.org): OpenML is an online platform that hosts a vast collection of datasets for machine learning research. It also provides tools and resources for collaborative machine learning experimentation.

# ### 4.1 Reading data from the internet
# 
# You can pull data directly from the internet using the pandas different read functions. For example `read_html` can pull data from tables. For example, we can pull the [table of all the physical constants from Wikipedia](https://en.wikipedia.org/wiki/List_of_physical_constants) using just a couple lines of Python code.

# In[10]:


url = 'https://en.wikipedia.org/wiki/List_of_physical_constants'
tables = pd.read_html(url)
df = tables[0]

df.head()


# <font size=+3>&#9998;</font> Do This - Find a physics data set of interest and read it in with pandas (from the web). Make plots of the data contained in the file.

# In[10]:


# Put your code here


# ---
# ### &#128721; You made it to the end!
# Is there anything that we covered today that you're still not feeling certain about? Talk to your group or check in with an instructor.
# 
# ---
