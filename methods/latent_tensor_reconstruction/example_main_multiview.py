## LTR example main file 
## data generation by random polynomial function
import sys
import time

import numpy as np
import scipy.stats as spstats 

## ###################################################
## for demonstration
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmplot
import sklearn.model_selection 
from sklearn.metrics import mean_squared_error, mean_absolute_error
## ###################################################

import ltr_solver_multiview_010 as ltr
import data_load_cls
sys.path.append('../util')
from util import load_csv_file_into_variable, save_array_to_file
from pathlib import Path
from decimal import *

all_rmse = []
all_mae = []
all_pcorr = []

## ################################################################
## ################################################################
## ###################################################
def acc_eval(yobserved, ypredicted, ypredictedrounded):
  """
  Task: to report some statistics, 
        f1, precision and recall is meaningfull if the data is binary coded (1,0,-1). 

  Input: yobserved     2d array of observed data
         ypredicted    2d array of predicted data
         ibinary       binary(=0,1) =1 binary coded data (=1,0,-1), =0 rela values 
  Output: pcorr        Pearson correlation
          rmse         Root mean square error
          prec         precision    
          recall       recall        
          f1           f1 measure
  """

  ## nobject,nclass=yobserved.shape

  tp=np.sum((yobserved>0)*(ypredicted>0))     ## true positive
  fp=np.sum((yobserved<=0)*(ypredicted>0))    ## false positive
  fn=np.sum((yobserved>0)*(ypredicted<=0))    ## false negative
  tn=np.sum((yobserved<=0)*(ypredicted<=0))   ## true negative

  print('Tp,Fn,Fp,Tn:',tp,fn,fp,tn)

  if tp+fp>0:
    prec=tp/(tp+fp)
  else:
    prec=0
    
  if tp+fn>0:
    recall=tp/(tp+fn)
  else:
    recall=0
  
  if prec+recall>0:
    f1=2*prec*recall/(prec+recall)
  else:
    f1=0
  
  ## Pearson correlation
  pcorr = np.corrcoef(yobserved.ravel(), ypredicted.ravel())[0, 1]
  ## Root mean sqaure error
  scorr = spstats.spearmanr(yobserved.ravel(), ypredictedrounded.ravel())[0]
  rmse = np.sqrt(mean_squared_error(yobserved.ravel(), ypredictedrounded.ravel()))
  mae = mean_absolute_error(yobserved.ravel(), ypredictedrounded.ravel())
  
  return(pcorr, scorr, rmse, mae)

## ################################################################
def print_result(xstat):
  """
  xstat = (pcorr,scorr,rmse,precision,recall,f1)
  """
  
  nitem = len(xstat)
  stext = ['P-corr','RMSE','Precision','Recall','F1']
  if len(stext) == nitem:
    for i in range(nitem):
      print(stext[i]+':','%7.4f'%(xstat[i]))

  return

## ################################################################
def main(iworkmode=None):
  """
  Task: to run the LTR solver on randomly generated polynomial functions
  """
  
  ## -------------------------------------
  ## Parameters to learn
  ## the most important parameter
  norder=2      ## maximum power
  rank=100      ## number of rows
  rankuv=50   ## internal rank for bottlenesck if rankuv<rank
  sigma=0.01  ## learning step size
  nsigma=10      ## step size correction interval
  gammanag=0.99     ## discount for the ADAM method
  gammanag2=0.99    ## discount for the ADAM method norm

  # mini-bacht size,
  mblock=500

  ## number of epochs
  nrepeat=20

  ## regularizatin constant for xlambda optimization parameter
  cregular=0.1  

  ## activation function
  iactfunc = 0  ## =0 identity, =1 arcsinh, =2 2*sigmoid-1, =3 tanh, =4 relu

  ## cmodel.lossdegree = 0  ## =0 L_2^2, =1 L^2, =0.5 L_2^{0.5}, ...L_2^{z}
  lossdegree=0  ## default L_2^2 =0
  regdegree=1   ## regularization degree, Lasso

  norm_type  = 0 ## parameter normalization =0 L2 =1 L_{infty} =2 arcsinh + L2 
                 ## =3 RELU, =4 tanh + L_2 

  perturb = 0 ## gradient perturbation

  report_freq = 50 ## frequency of the training reports

  ## --------------------------------------------
  cmodel=ltr.ltr_solver_cls(norder=norder,rank=rank,rankuv=rankuv)

  ## set optimization parameters
  cmodel.update_parameters(nsigma=nsigma, \
                           mblock=mblock, \
                           sigma0=sigma, \
                           gammanag=gammanag, \
                           gammanag2=gammanag2, \
                           cregular=cregular, \
                           iactfunc=iactfunc, \
                           lossdegree=lossdegree, \
                           regdegree=regdegree, \
                           norm_type =norm_type, \
                           perturb = perturb, \
                           report_freq = report_freq)
                           
  print('Order:',cmodel.norder)
  print('Rank:',cmodel.nrank)
  print('Rankuv:',cmodel.nrankuv)
  print('Step size:',cmodel.sigma0)
  print('Step freq:',cmodel.nsigma)
  print('Step scale:',cmodel.dscale)
  print('Epoch:',nrepeat)
  print('Mini-batch size:',mblock)
  print('Discount:',cmodel.gamma)
  print('Discount for NAG:',cmodel.gammanag)
  print('Discount for NAG norm:',cmodel.gammanag2)
  print('Bag size:',cmodel.mblock)
  print('Regularization:',cmodel.cregular)
  print('Gradient max ratio:',cmodel.sigmamax)
  print('Type of activation:',cmodel.iactfunc)
  print('Degree of loss:',cmodel.lossdegree)
  print('Degree of regularization:',cmodel.regdegree)
  print('Normalization type:',cmodel.norm_type)
  print('Gradient perturbation:', cmodel.perturb)
  print('Activation:', cmodel.iactfunc)
  print('Input centralization:', cmodel.ixmean)
  print('Input L_infty scaling:', cmodel.ixscale)
  print('Quantile regression:',cmodel.iquantile)
  print('Quantile alpha:',cmodel.quantile_alpha)    ## 0.5 for L_1 norm loss
  print('Quantile smoothing:',cmodel.quantile_smooth)

  #####################################
  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  ## Data loading

  ## data directory
  sdir = '../../dataset/weekly/'
  
  # Test cases
  known_weeks_test_cases = [11, 8, 5, 3]
  test_case_nr = 1
  for number_of_weeks_with_known_data in known_weeks_test_cases:
      print("---- Starting LTR for ", number_of_weeks_with_known_data, " known weeks")
      for percentage_of_students_missing in ["30", "60", "90"]: 
          print(percentage_of_students_missing, " % missing ")
          # Create directories for saving results
          Path("../results/test-case-" + str(test_case_nr) + "/" + percentage_of_students_missing + "p-students-missing/ltr").mkdir(parents=True, exist_ok=True)
          
          test_student_ids = load_csv_file_into_variable('../../results/test-student-ids-' + percentage_of_students_missing + '.csv')
          cdata = data_load_cls.data_cls(sdir)
          cdata.load_files(test_student_ids[0], number_of_weeks_with_known_data)

          ## data tables
          xweek = cdata.xweek
          xstudent = cdata.xstudent
          xerror = cdata.xerror
          xindextrain = cdata.xindextrain  ## the join table
          xindextest = cdata.xindextest  ## the join table
          ytrain = cdata.ytrain
          ytest = cdata.ytest

          save_array_to_file(ytrain, '../results/test-case-' + str(test_case_nr) + '/' + percentage_of_students_missing + 'p-students-missing/ltr/Y_train.csv')
          save_array_to_file(ytest, '../results/test-case-' + str(test_case_nr) + '/' + percentage_of_students_missing + 'p-students-missing/ltr/real-values.csv')


          ## training
          ## some design configuration
          idesign  = 0   ## some designs are demonstrated here
          time0=time.time()
          
          if idesign == 0: ## simplest case
            ## views: weeks, students, errors
              ## the order of the model, polynomial 3 
              llinks = [0, 1, 2] #=> xyx
              lxtrain = [xweek, xstudent, xerror]

              xindex_train = xindextrain
              xindex_test = xindextest

              cmodel.fit(lxtrain, ytrain, llinks = llinks, xindex = xindex_train, \
                                  nepoch=nrepeat)
              time1 = time.time()

              ## prediction
              lxtest = lxtrain
              ypred_test = cmodel.predict(lxtest, llinks = llinks, \
                                                xindex = xindex_test)

          print('Training time:', time1 - time0)

          # save the prediction
          prediction = ypred_test.ravel()
          save_array_to_file(prediction, '../results/test-case-' + str(test_case_nr) + '/' + percentage_of_students_missing + 'p-students-missing/ltr/predicted-values.csv')
          prediction_rounded = np.asarray(list(map(lambda x : abs(float(round(Decimal(str(x)), 0))), prediction)))          
          save_array_to_file(prediction_rounded, '../results/test-case-' + str(test_case_nr) + '/' + percentage_of_students_missing + 'p-students-missing/ltr/predicted-values-rounded.csv')
          results = acc_eval(ytest.ravel(), prediction.ravel(), prediction_rounded.ravel())
          print('Prediction accuracy: ', results)
          
          all_pcorr.append(results[0])
          all_rmse.append(results[2])
          all_mae.append(results[3])

      test_case_nr += 1
    

  print("RMSE: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_rmse)))
  print("MAE: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_mae)))
  print("PCOR: ", list(map(lambda x : abs(float(round(Decimal(str(x)), 4))), all_pcorr)))
  print('Bye')

  return(0)

## ###################################################
## ################################################################
if __name__ == "__main__":
  if len(sys.argv)==1:
    iworkmode=0
  elif len(sys.argv)>=2:
    iworkmode=eval(sys.argv[1])
  main(iworkmode)
