import numpy
from util.util import save_dataset_to_file, load_csv_file_into_variable, calculate_RMSE
from soft_impute_impl.soft_impute import SoftImpute
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from decimal import *


all_rmse = []
all_mae = []
all_pcorr = []

def generate_mask(number_of_weeks_with_known_data, test_students_ids):
    train_mask1 = generate_mask_for_known_part_of_the_semester(number_of_weeks_with_known_data)
    train_mask2 = generate_mask_for_unknown_part_of_the_semester(22 - number_of_weeks_with_known_data, test_students_ids)
    mask = numpy.append(train_mask1, values=train_mask2, axis=1)
    return mask

# Function for generating first part of the mask - known data
# Parameters: int-> number of weeks with known data
# Returns: array -> first part of the mask
def generate_mask_for_known_part_of_the_semester(number_of_weeks_with_known_data):
    return numpy.full(shape=(439, number_of_weeks_with_known_data*20), fill_value=False)

# Function for generating the second part of the mask that contains some missing values
# Parameters: int -> number of weeks with some missing data
# Returns: array -> second part of the mask, contains True for test student ids, and False for other students
def generate_mask_for_unknown_part_of_the_semester(number_of_weeks_with_missing_data, test_students_ids):
    mask = numpy.full(shape=(439, number_of_weeks_with_missing_data*20), fill_value=False)
    for test_student_id in test_students_ids:
        # For this student id, set all values for the last weeks that are missing to True
        mask[test_student_id, :] = True
    return mask


def generate_train_dataset(dataset, test_students_ids, number_of_weeks_with_known_data):
    mask = generate_mask(number_of_weeks_with_known_data, test_students_ids)
    train_dataset = numpy.copy(dataset)
    train_dataset[mask] = numpy.nan
    return train_dataset

def perform_soft_iterative_imputation(train_dataset, number_of_weeks_with_known_data, percentage_of_students_missing, test_student_ids, dataset, test_case_nr):
    clf = SoftImpute()
    clf.fit(dataset)
    train_dataset_after_soft_imputation = clf.predict(train_dataset)

    # Get predicted and real values    
    predicted_values = []
    real_values = []
    for test_student_id in test_student_ids:
        # Predicted values are values on same positions as real but from the transformed dataset
        predicted_values.append(train_dataset_after_soft_imputation[test_student_id, number_of_weeks_with_known_data*20:])
        # Real values are values from the original matrix on positions 
        # row = test student id 
        # column = all starting from number_of_weeks_with_known_data*20 until the end
        real_values.append(dataset[test_student_id, number_of_weeks_with_known_data*20:])
    predicted_values = numpy.asarray(predicted_values)
    save_dataset_to_file(predicted_values, 'results/test-case-' + test_case_nr +'/'+ percentage_of_students_missing + 'p-students-missing/svd-based/predicted-values.csv')

    # Round predicted values
    predicted_values_rounded = numpy.copy(predicted_values)
    for i in range(0, predicted_values_rounded.shape[0]):
        predicted_values_rounded[i, :] = list(map(lambda x : abs(float(round(Decimal(str(x)), 0))), predicted_values_rounded[i, :]))
    
    save_dataset_to_file(predicted_values_rounded, 'results/test-case-' + test_case_nr +'/'+ percentage_of_students_missing + 'p-students-missing/svd-based/predicted-values-rounded.csv')
    real_values = numpy.asarray(real_values)
    save_dataset_to_file(real_values, 'results/test-case-' + test_case_nr +'/'+ percentage_of_students_missing + 'p-students-missing/svd-based/real-values.csv')

    rmse_ice = calculate_RMSE(real_values.ravel(), predicted_values_rounded.ravel())
    print('Test case ' , test_case_nr, ".", percentage_of_students_missing)
    print("RMSE: {:.4f}".format(rmse_ice))
    all_rmse.append(rmse_ice)
    mae = mean_absolute_error(real_values.ravel(), predicted_values_rounded.ravel())
    print("MAE: {:.4f}".format(mae))
    all_mae.append(mae)
    pcorr = numpy.corrcoef(real_values.ravel(), predicted_values.ravel())[0, 1]
    print("PCORR on test set: {:.4f}".format(pcorr))
    all_pcorr.append(pcorr)



def main():
    # Define ROUND_HALF_UP as rounding strategy
    getcontext().rounding = ROUND_HALF_UP

    known_weeks_test_cases = [11, 8, 5, 3]
    test_case_nr = 1
    for number_of_weeks_with_known_data in known_weeks_test_cases:
        print("---- Starting soft iterative imputation for ", number_of_weeks_with_known_data, " known weeks")
        for percentage_of_students_missing in ["30", "60", "90"]:
            print(percentage_of_students_missing, " % missing ")
            # Create directories for saving results
            Path("results/test-case-" + str(test_case_nr) + "/" + percentage_of_students_missing + "p-students-missing/svd-based").mkdir(parents=True, exist_ok=True)

            test_student_ids = load_csv_file_into_variable('../results/test-student-ids-' + percentage_of_students_missing + '.csv')[0]
            test_student_ids = numpy.asarray(test_student_ids, dtype='int64')
            # Load dataset that is in form of a 2D matrix of size (nr_of_students, nr_of_errors*nr_of_weeks) -> (439, 440)
            dataset = load_csv_file_into_variable('../dataset/dataset-2d-matrix.csv')
            train_dataset = generate_train_dataset(dataset, test_student_ids, number_of_weeks_with_known_data)
            save_dataset_to_file(train_dataset, 'results/test-case-' + str(test_case_nr) +'/' + percentage_of_students_missing + 'p-students-missing/svd-based/train-dataset.csv')
            perform_soft_iterative_imputation(train_dataset, number_of_weeks_with_known_data, percentage_of_students_missing, test_student_ids, dataset, str(test_case_nr))
        test_case_nr += 1
    
    print("RMSE: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_rmse)))
    print("MAE: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_mae)))
    print("PCOR: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_pcorr)))


if __name__ == '__main__':
    main()
