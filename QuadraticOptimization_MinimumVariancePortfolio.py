# -*- coding: utf-8 -*-
"""
Created on Sat April 17 09:58:05 2019

@author: Ray Chen

Title: Mean Variance Portfolio Family 
Including: minimum variance, maximum sharpe ratio and mean variance
"""
import numpy as np
import pandas as pd 
import quadprog as qp

################################################################################
# Function 
################################################################################
# This is self defined 
def optimize(dt):
    # input a dataframe with all asset returns
    
    # This is the optimization
    # min 1/2 x^t*G*x - a^T*X
    # C^t * X >= b
    
    #------------------------------------------------- Basic Setting
    # Set minimum and Maximum Allocation weight for each position
    min_allo = 0.05
    max_allo = 0.60
    P = np.array(dt.cov())
    
    try:  # make sure P is positive defined
        qp_G = .5 * (P + P.T)
    except ValueError:        
        print("input covariance matrix is not positive defined")
    
    # Expected Return from historical mean
    qp_a = np.array(dt.apply(lambda col: col.mean()))
    dm = qp_G.shape[0]
    
    # set constraint matrix
    qp_C = np.vstack([np.eye(dm, dm), -np.eye(dm, dm)])
    qp_C = np.vstack([np.tile(1, dm), qp_C])


    b_1 = np.array([1])
    # set minimum and maximum range
    b_min = np.tile(min_allo, qp_G.shape[1])
    b_max = np.tile(-max_allo, qp_G.shape[1])

    qp_b = np.append(b_1, b_min)
    qp_b = np.append(qp_b, b_max)
    
    # Specify number of equality constraints
    meq = 1

    #------------------------------------------------- Optimization
    # If it is minimum variance, qp_a_tp = [0,0,..., 0]
    qp_a_tp = np.array([qp_a * 0])
   
    # If it is mean variance, qp_a_tp = qp_a, realized mean return from lookback periods
    # qp_a_tp = np.array([qp_a * 1])

    ### Be careful on np.array's dimension
    solved_result = qp.solve_qp(qp_G, qp_a_tp[0,], qp_C.T, qp_b, meq)
    
    wt = np.array([solved_result[0]]).T
    
    # Portfolio Return = np.dot(qp_a_tp, wt)[0, 0]
    # Portfolio sd = (np.dot(wt.T, np.dot(qp_G, wt)) ** (0.5))[0, 0]
    
    # transpose weight
    wt_ts = wt.T
  
    return wt_ts


