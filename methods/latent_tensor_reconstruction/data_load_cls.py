import sys
import time
import os
import csv
## import pylab as plt

import numpy as np

## ########################################################

## #############################################################
## #############################################################
## #############################################################
## real encoding <-> density coding
## #############################################################
class data_cls:

  def __init__(self,sdir):

    self.sdir = sdir    ## directory of raw data
    self.ytrain = None         ## counts ordered to the join index
    self.ytest = None
    self.xindextrain = None    ## indeces for  tensor join
    self.xindextest = None
    self.xstudent = None  ## student indicators 
    self.xerror = None    ## error indicators 
    self.xweek = None     ## vecot of the weeks

    self.xtensor = None  ## full, 3 order, data tensor: week,student,error -> count
    self.dstudent = {}  ## student id : student index
    self.derror = {}    ## error name : error index
    
    return

  ## ----------------------------------------------------------
  def load_files(self, test_student_ids, number_of_weeks_with_known_data, header = 'week', nfile = 22):
    """
    Task is to load the raw error score files
    Input: header  string of the file header:  filename headrer+str(i).csv
           nfile   number of files i =1,...,n
    Output: see object variables
    """

    self.xweek = np.arange(nfile)

    ## count the rows and column

    ## indexes of students and errors
    istudent = 0
    ierror =0
    sstudent = set([])
    
    ifirstfile = 1
    for ifile in range(nfile):
      fname = self.sdir + header + str(ifile+1) + '.csv'
      with open(fname,newline='') as csvfile:
        scorereader = csv.reader(csvfile,delimiter = ',')
        nline = 0
        ifirstline = 1
        for line in scorereader:
          if ifirstline == 1 and ifirstfile == 1:
            ncolumn = len(line)
            ifirstfile = 0
            ifirstline = 0
            self.derror = { line[i+1]: i for i in range(ncolumn-1)}
          elif ifirstline == 0:
            student_id = line[0]
            if student_id not in self.dstudent:
              sstudent.add(student_id)
          nline += 1

    lstudent = list(sstudent)
    self.dstudent = { lstudent[i] : i for i in range(len(lstudent))}
          
    nstudent = len(self.dstudent)
    nweek = nfile
    nerror = ncolumn -1
        
    nstudenttest = len(test_student_ids)
    self.xindextest = np.zeros(((22-number_of_weeks_with_known_data)*nstudenttest*nerror,3), dtype=int)
    # Train dataset has data from all weeks for train students and number_of_weeks_with_known_data for test students
    self.xindextrain = np.zeros((nweek*(439-nstudenttest)*nerror + number_of_weeks_with_known_data*nstudenttest*nerror,3), dtype=int)
    self.xtensor = np.zeros((nweek,nstudent,nerror))
    self.xstudent = np.eye(nstudent)
    self.xerror = np.eye(nerror)
    self.xweek = np.arange(nweek, dtype=float)
    self.ytest = np.zeros((22-number_of_weeks_with_known_data) * nstudenttest * nerror)
    # Train dataset has data from all weeks for train students and number_of_weeks_with_known_data for test students
    self.ytrain = np.zeros(nweek * (439-nstudenttest) * nerror + number_of_weeks_with_known_data*nstudenttest*nerror)

    ixtrain = 0
    ixtest = 0
    for ifile in range(nfile):
      fname = self.sdir + header + str(ifile+1) + '.csv'
      with open(fname,newline='') as csvfile:
        scorereader = csv.reader(csvfile,delimiter = ',')
        nline = 0
        for line in scorereader:
          if nline >= 1:
            xrow = np.array(line[1:])
            # ifile is the week number
            # nline-1 is our new student id
            if ifile >= number_of_weeks_with_known_data and nline-1 in test_student_ids:
              self.ytest[ixtest:ixtest+nerror] = xrow
              self.xindextest[ixtest:ixtest+nerror,0] = ifile
              self.xindextest[ixtest:ixtest+nerror,1] = nline-1
              self.xindextest[ixtest:ixtest+nerror,2] = np.arange(nerror)
              self.xtensor[ifile,nline-1,:] = xrow
              ixtest += nerror
            else:
              self.ytrain[ixtrain:ixtrain+nerror] = xrow
              self.xindextrain[ixtrain:ixtrain+nerror,0] = ifile
              self.xindextrain[ixtrain:ixtrain+nerror,1] = nline-1
              self.xindextrain[ixtrain:ixtrain+nerror,2] = np.arange(nerror)
              self.xtensor[ifile,nline-1,:] = xrow
              ixtrain += nerror
          nline +=1
          ## print(ix)
              
    return

## ################################################################
def main(iworkmode):

  sdir = '/home/sandor/data/student_error/'
  cdata = data_cls(sdir)
  cdata.load_files()

  print('Bye')
  
  return

## ################################################################
if __name__ == "__main__":
  if len(sys.argv)==1:
    iworkmode=0
  elif len(sys.argv)>=2:
    iworkmode=eval(sys.argv[1])
  main(iworkmode)

