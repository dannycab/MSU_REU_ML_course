# Independent Project Work

Now that you have had a some time to work with the `scikit` workflow and see the differences in the main classes of problems we solve with machine learning, you will have the opportunity to work on a project of your own design. This is completely optional, but highly encouraged. You will not learn how to work with machine learning tools without practice and this is a great opportunity to get some practice. Below is a list of criteria that you should follow when designing your project. These will help you to get the most out of the experience.

## Classification Problem

A classification problem will use already tagged data (supervised learning) to predict the category of new observations. Here you might need to look for data or ask your research group if such data might exist for the problem you are working on. We recommend that you work with a data set that has at least 1000 observations or more. Training a model on a small data set can lead to overfitting and poor generalization. It would be good the number of features to be about 10 or more. This will give you a good sense of how to work with a data set that is not too small or too large. Having two many features would suggest that we instead use something like [dimension reduction](https://www.wikipedia.org/wiki/Dimensionality_reduction) to reduce the number of features before modeling -- this is good practice because big data sets can be computationally expensive to work with.

To successfully complete a classification problem, you should be able to:
* Read data into a pandas DataFrame
* Clean data and impute data (as needed) - we didn't do much imputation in this course, but it is a common practice in machine learning
* Plot data and propose potential models for analysis
* Split data into training and testing sets
* Build classification models
* Evaluate the quality of fitted models (here you might use a confusion matrix, ROC curve, or other metrics)

## Regression Problem

A regression problem will use data to predict a continuous outcome. Again, you might need to look for data or ask your research group if there's some data that might exist for you. We recommend that you work with a data set that has at least 1000 observations or more -- for the same reasons as above. The number of features should be about 10. Try not to work with too many features - again, for the same reasons as above.

To successfully complete a regression problem, you should be able to:
* Read data into a pandas DataFrame
* Clean data and impute data (as needed)
* Plot data and propose potential models for analysis
* Split data into training and testing sets
* Build regression models
* Evaluate the quality of the fit using metrics like the mean squared error, R^2, etc.


### Physics Data Sets

We used the famous Iris data set to do this activity. That is because it is well known and clearly documented. Most data sets you will work with will not have the same level of documentation. Let's try to find some physics data sets that you can read in and plot.

Places to look for data: 

* [Kaggle](https://www.kaggle.com): Kaggle is a well-known platform for data science and machine learning. It offers a vast collection of datasets contributed by the community. You can search for datasets based on various categories, such as image data, text data, time series data, etc.
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php): The UCI Machine Learning Repository is a popular resource for machine learning datasets. It provides a diverse collection of datasets from various domains, including classification, regression, clustering, and more.
* [Google Dataset Search](https://datasetsearch.research.google.com): Google Dataset Search allows you to search for datasets across different domains. It aggregates datasets from various sources, including academic institutions, data repositories, and other websites.
* [Data.gov](https://www.data.gov): Data.gov is the U.S. government's open data portal. It provides access to a wide range of datasets covering various topics, such as health, climate, transportation, and more. It's a valuable resource for finding government-related datasets.
* [Microsoft Research Open Data](https://msropendata.com): Microsoft Research Open Data is a platform that provides access to diverse datasets collected or curated by Microsoft researchers. It includes datasets from domains like computer vision, natural language processing, and healthcare.
* [AWS Open Data Registry](https://registry.opendata.aws): AWS Open Data Registry is a collection of publicly available datasets provided by Amazon Web Services (AWS). It hosts a variety of datasets, including satellite imagery, genomics data, and more.
* [OpenML](https://www.openml.org): OpenML is an online platform that hosts a vast collection of datasets for machine learning research. It also provides tools and resources for collaborative machine learning experimentation.