# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Wed Nov  6 22:46:40 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student
"""
import bayes_opt as bayesO

class BayesOpt():
    
    def __init__(self,
                 function,
                 pbounds,
                 no_iter = 5,
                 start_p = 5):   #Starting Points
        
        self.function = function
        self.pbounds  = pbounds
#        self.no_iter  = no_iter
#        self.start_p  = start_p
        
        self.opt = bayesO.BayesianOptimization(f       = self.function,
                                               pbounds = self.pbounds,
                                               random_state=1)
      
        self.opt.maximize(init_points = no_iter,
                          n_iter      = start_p)
        
    def resume(self,
               no_iter = 5,
               start_p = 0):
        
        self.opt.maximize(init_points = no_iter,
                          n_iter      = start_p)
        
class PSO():
    
    pass
        
        
        
        
    
    