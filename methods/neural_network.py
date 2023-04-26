import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Concatenate
from keras.utils.np_utils import to_categorical
from util.util import root_mean_squared_error, calculate_RMSE, save_dataset_to_file, save_array_to_file, load_csv_file_into_variable
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from decimal import *

all_rmse = []
all_mae = []
all_pcorr = []

def load_values_from_dataset():
     errorDataset=pd.read_excel(
          r"../dataset/Dataset.xlsx",
     )
     errorDataset.head()
     allValues = errorDataset.iloc[:,:].values
     
     X1 = allValues[:, 0] # feature 1 - student
     X2 = allValues[:, 1] # feature 2 - error
     X3 = allValues[:, 2] # feature 3 - week
     Y = allValues[:, 3] # output - number of occurrences
     Y = Y.reshape(-1, 1)

     X1 = X1.reshape(-1, 1)
     X1 = to_categorical(X1)
     X2 = X2.reshape(-1, 1)
     X2 = to_categorical(X2)
     X3 = X3.reshape(-1, 1)
     X3 = to_categorical(X3)

     X = np.zeros((len(X1), 439 + 20 + 22))
     X[:, 0:439] = X1
     X[:, 439:459] = X2
     X[:, 459:] = X3

     return X, Y

def generate_train_and_test_data(test_student_ids, number_of_weeks_with_known_data, test_case_nr, percentage_of_students_missing):
     X, Y = load_values_from_dataset()
     X_test = []
     X_train = []
     Y_test = []
     Y_train = []
     train_dataset = []

     for i in range(len(X)):
          student = X[i, 0:439]
          error = X[i, 439:459]
          week = X[i, 459:]
          nr_of_occurences = Y[i]
          if np.argmax(student) in test_student_ids and np.argmax(week) in range(number_of_weeks_with_known_data, 23):
               X_test.append(X[i])
               Y_test.append(nr_of_occurences)
          else:
               X_train.append(X[i])
               Y_train.append(nr_of_occurences)
               train_dataset.append([np.argmax(student), np.argmax(error), np.argmax(week), nr_of_occurences[0]])

     X_train = np.asarray(X_train)
     X_test = np.asarray(X_test)
     Y_train = np.asarray(Y_train, dtype='float32').reshape(-1)
     Y_test = np.asarray(Y_test, dtype='float32').reshape(-1)

     save_array_to_file(Y_test, 'results/test-case-' + test_case_nr + '/' + percentage_of_students_missing + 
                           'p-students-missing/neural-network/real-values.csv')
     save_dataset_to_file(np.asarray(train_dataset), 'results/test-case-' + test_case_nr + '/' + percentage_of_students_missing + 
                           'p-students-missing/neural-network/train-dataset.csv')
     
     return X_train, X_test, Y_train, Y_test

def create_and_train_neural_network(X_train, Y_train):
     studentInput = Input(shape=(439,))
     compilerErrorInput = Input(shape=(20,))
     weekInput = Input(shape=(22,))

     x = Concatenate()([studentInput, compilerErrorInput, weekInput])
     x = Dense(64, kernel_initializer='normal', activation='sigmoid')(x)
     x = Dense(1, kernel_initializer='normal', activation='relu')(x)
     model = Model([studentInput, compilerErrorInput, weekInput], x)
     model.compile(optimizer='Adam', loss=root_mean_squared_error, metrics=['accuracy'])
     model.fit([X_train[:, 0:439], X_train[:, 439:459], X_train[:, 459:]], Y_train, batch_size=200, epochs=20)

     return model

def predict_values(model, X_test, Y_test, test_case_nr, percentage_of_students_missing):
     score = model.evaluate([X_test[:, 0:439], X_test[:, 439:459], X_test[:, 459:]], Y_test, verbose=0)
     print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

     Y_pred = model.predict([X_test[:, 0:439], X_test[:, 439:459], X_test[:, 459:]]).reshape(-1)
     save_array_to_file(Y_pred, 'results/test-case-' + test_case_nr + '/' + percentage_of_students_missing + 
                           'p-students-missing/neural-network/predicted-values.csv')
     
     # Round values
     Y_pred_rounded = np.asarray(list(map(lambda x : abs(float(round(Decimal(str(x)), 0))), Y_pred)))
     save_array_to_file(Y_pred_rounded, 'results/test-case-' + test_case_nr + '/' + percentage_of_students_missing + 
                           'p-students-missing/neural-network/predicted-values-rounded.csv')
     print(test_case_nr, '.', percentage_of_students_missing, '.' )
     all_rmse.append(calculate_RMSE(Y_test, Y_pred_rounded))
     all_mae.append(mean_absolute_error(Y_test, Y_pred_rounded))
     all_pcorr.append(np.corrcoef(Y_test, Y_pred)[0, 1])


def main():
     # Define ROUND_HALF_UP as rounding strategy
     getcontext().rounding = ROUND_HALF_UP

     known_weeks_test_cases = [11, 8, 5, 3]
     test_case_nr = 1
     for number_of_weeks_with_known_data in known_weeks_test_cases:
          print("---- Starting NN for ", number_of_weeks_with_known_data, " known weeks")
          for percentage_of_students_missing in ["30", "60", "90"]:
               print(percentage_of_students_missing, " % missing ")
               # Create directories for saving results
               Path("results/test-case-" + str(test_case_nr) + "/" + percentage_of_students_missing + "p-students-missing/neural-network").mkdir(parents=True, exist_ok=True)

               test_student_ids = load_csv_file_into_variable('../results/test-student-ids-' + percentage_of_students_missing + '.csv')
               X_train, X_test, Y_train, Y_test = generate_train_and_test_data(test_student_ids, number_of_weeks_with_known_data, str(test_case_nr), percentage_of_students_missing)
               model = create_and_train_neural_network(X_train, Y_train)
               predict_values(model, X_test, Y_test, str(test_case_nr), percentage_of_students_missing)
          
          test_case_nr += 1
     
     print("RMSE: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_rmse)))
     print("MAE: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_mae)))
     print("PCOR: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_pcorr)))
     
     

if __name__ == "__main__":
     main()
