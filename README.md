# DM_project - Customer Segmentation

| Grade                 | Mean                  | Q75                |
|:---------------------:|:---------------------:|:------------------:|
| 19.9/20 (best project)| 14.6                  | 17.5               |

The client, an insurance company, wishes to better understand the scope of its clients, in order to better serve them and increase their ROI (Return On Investment). The group was given an ABT (Analytic Based Table), consisting of 10.290 customers and given the task of analyzing the table for evident groups of clusters, extracting the behavior of said clusters and provide insights on how to better serve them. 

This was an end-to-end project which means that we had to gather some business understanding about the insurance sector, followed by understanding the data and preparing it (outliers, missing values, incoherencies) so we could then cluster the clients based on their characteristics according to two main perspectives: value and product. We tried several different models with different parameter choices for each view and then we selected the most appropriate one after comparing them on cluster quality measures. We also characterized each cluster by using visual methods and proposed a marketing strategy for each segment.

Finally, we proposed a single set of clusters instead of the ones of each perspective and we built a simple decision tree classifier to allow future customer classification and also to understand what are the most distinguishing characteristics across clusters.

The report which accompanies the notebook can be consulted at: https://tinyurl.com/dm-segmentation.


# Setup Environment
Before running this notebook, the user needs to configure a separate conda environment in order to reproduce the results obtained. An environment.yml file is provided in the project repository with this objective. By creating a conda environment through the yml file, the user will be able to run the whole project without issues. Besides, the user also needs to install an additional package from a personal github repository since it is not available in pypi nor in conda repositories.

In order to create the environment the user needs to run the following commands in the anaconda prompt: **conda env create -f \<path to environment.yml>**. Afterwards, the user needs to activate the created environment: **conda activate datamin** and initialize the jupyter notebook: **jupyter notebook**.

The final step is to install the sompy package from the personal github repository, which can be done by running the appropriate cell in the beginning of the Notebook.

