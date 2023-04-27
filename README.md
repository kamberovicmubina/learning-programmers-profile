# learning-programmers-profile

This repository contains supporting materials from submission for UMAP 2023.

## Repository structure

- **dataset** - contains three forms of dataset:
  - Weekly data - one csv file for each week of the semester, where every line represents the number of occurences of the 20 compiler errors that one student made in that week.
  - Dataset in form of an excel file with 4 columns: student, error, week and number of occurences.
  - Weekly data narrowed down into one file, where every line is the concatenated weekly data for one student (i.e. the first 20 values in a row represent the number of occurences that student made in the first week, the next 20 values are related to the second week etc).


- **methods** - contains scripts for running the following methods on the dataset: svd-based, latent tensor reconstruction, neural network and gradient boosting.

Note: Upon script execution, a results folder will be generated, containing all predicted and actual values for each test case and each method.


- **results** - contains files with student ids that were used as test data, and a zip file with the predicted and actual values on which the methods were evaluated.
