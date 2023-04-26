import numpy
import csv
import math
import os
from keras import backend as K
from sklearn.metrics import mean_squared_error

def save_dataset_to_file(dataset, fileName):
    with open(fileName, "a", newline="") as csvFile:
        writer = csv.writer(csvFile)
        for i in range(0, dataset.shape[0]):
            row = []
            for j in range(0, dataset.shape[1]):
                row.append(dataset[i][j])
            writer.writerow(row)

def save_array_to_file(custom_array, fileName):
    with open(fileName, "a", newline="") as csvFile:
        csv.writer(csvFile).writerow(custom_array)

def load_csv_file_into_variable(fileName):
    with open(fileName, "r", newline="") as csvFile:
        return numpy.asarray(list(csv.reader(csvFile, quoting=csv.QUOTE_NONNUMERIC)))

def calculate_RMSE(original_values, predicted_values):
    return math.sqrt(mean_squared_error(original_values, predicted_values))

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
