# DataMiningFinalProject


The client, an insurance company, wishes to better understand the scope of its clients, in order to better serve them and increase their ROI (Return On Investment).
The group was given an ABT (Analytic Based Table), consisting of 10.290 customers and given the task of analyzing the table for evident groups of clusters, extracting the behaviour of said clusters and provide insights on how to better serve them. 


## 1.1: Setup Environment
Before running this notebook, the user needs to configure a separate conda environment in order to reproduce the results obtained. An environment.yml file is provided in the project repository with this objective. By creating a conda environment through the yml file, the user will be able to run the whole project without issues. Besides, the user also need to install an additional package from a personal github repository since it is not available in pypi nor in conda repositories.

In order to create the environment the user needs to run the following commands in the anaconda prompt: **conda env create -f \<path to environment.yml>**. Afterwards, the user needs to activate the created environment: **conda activate datamin** and initialize the jupyter notebook: **jupyter notebook**.

The final step is to install the sompy package from the personal github repository, which can be done by running the cell below.
