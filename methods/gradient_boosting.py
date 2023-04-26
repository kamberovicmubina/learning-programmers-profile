import pandas as pd
from util.util import calculate_RMSE, load_csv_file_into_variable, save_dataset_to_file, save_array_to_file
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from decimal import *

all_rmse = []
all_mae = []
all_pcorr = []

def generate_train_and_test_data(number_of_weeks_with_known_data, test_student_ids, test_case_nr, percentage_of_students_missing):
    errorDataset=pd.read_excel(
        r"../dataset/Dataset.xlsx",
    )
    errorDataset.head()
    allValues = errorDataset.iloc[:,:].values
    X = allValues[:, :-1]
    Y = allValues[:, 3:].ravel()

    X_test = []
    X_train = []
    Y_test = []
    Y_train = []
    train_dataset = []
    for i in range(len(X)):
        student = X[i, 0]
        error = X[i, 1]
        week = X[i, 2]
        nr_of_occurences = Y[i]
        if student in test_student_ids and week in range(number_of_weeks_with_known_data, 23):
            X_test.append(X[i])
            Y_test.append(nr_of_occurences)
        else:
            X_train.append(X[i])
            Y_train.append(nr_of_occurences)
            train_dataset.append([student, error, week, nr_of_occurences])

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    Y_train = np.asarray(Y_train, dtype='float32').reshape(-1)
    Y_test = np.asarray(Y_test, dtype='float32').reshape(-1)

    save_array_to_file(Y_test, 'results/test-case-' + test_case_nr + '/' + percentage_of_students_missing + 
                        'p-students-missing/gradient-boosting/real-values.csv')
    save_dataset_to_file(np.asarray(train_dataset), 'results/test-case-' + test_case_nr + '/' + percentage_of_students_missing + 
                        'p-students-missing/gradient-boosting/train-dataset.csv')
    
    return X_train, X_test, Y_train, Y_test


def main():
    # Define ROUND_HALF_UP as rounding strategy
    getcontext().rounding = ROUND_HALF_UP

    known_weeks_test_cases = [11, 8, 5, 3]
    test_case_nr = 1
    for number_of_weeks_with_known_data in known_weeks_test_cases:
        print("---- Starting GB for ", number_of_weeks_with_known_data, " known weeks")
        for percentage_of_students_missing in ["30", "60", "90"]:
            print(percentage_of_students_missing, " % missing ")
            # Create directories for saving results
            Path("results/test-case-" + str(test_case_nr) + "/" + percentage_of_students_missing + "p-students-missing/gradient-boosting").mkdir(parents=True, exist_ok=True)
            
            test_student_ids = load_csv_file_into_variable('../results/test-student-ids-' + percentage_of_students_missing + '.csv')
            X_train, X_test, Y_train, Y_test = generate_train_and_test_data(number_of_weeks_with_known_data, test_student_ids, str(test_case_nr), percentage_of_students_missing)
            gbr = GradientBoostingRegressor()
            gbr.fit(X_train, Y_train)
            pred_values = gbr.predict(X_test)
            save_array_to_file(pred_values, 'results/test-case-' + str(test_case_nr) + '/' + percentage_of_students_missing + 
                                'p-students-missing/gradient-boosting/predicted-values.csv')
            
            # Round values
            pred_values_rounded = np.asarray(list(map(lambda x : abs(float(round(Decimal(str(x)), 0))), pred_values)))
            save_array_to_file(pred_values_rounded, 'results/test-case-' + str(test_case_nr) + '/' + percentage_of_students_missing + 
                                'p-students-missing/gradient-boosting/predicted-values-rounded.csv')
            
            rmse = calculate_RMSE(Y_test, pred_values_rounded)
            print("RMSE on test set: {:.4f}".format(rmse))
            all_rmse.append(rmse)
            mae = mean_absolute_error(Y_test, pred_values_rounded)
            print("MAE on test set: {:.4f}".format(mae))
            all_mae.append(mae)
            pcorr= np.corrcoef(Y_test, pred_values)[0, 1]
            print("PCORR on test set: {:.4f}".format(pcorr))
            all_pcorr.append(pcorr)
        test_case_nr += 1

    print("RMSE: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_rmse)))
    print("MAE: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_mae)))
    print("PCOR: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_pcorr)))
    


if __name__ == '__main__':
    main()