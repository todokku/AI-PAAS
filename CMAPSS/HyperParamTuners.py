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
        self._no_iter  = no_iter
        self._start_p  = start_p
        
        self.opt = bayesO.BayesianOptimization(f       = self.function,
                                               pbounds = self.pbounds,
                                               random_state=1)
      
        self.opt.maximize(init_points = self._start_p,
                          n_iter      = self._no_iter)
        
    def __call__(self,
                 no_iter = None,
                 start_p = None):
        
        if no_iter is None:
            
            no_iter = self._no_iter
            
        if start_p is None:
            
            start_p = self._start_p
        
        self.opt.maximize(init_points = start_p,
                          n_iter      = no_iter)
        
class PSO():
    
    pass
        
        
if __name__ == '__main__':

    import numpy as np

    def ackley(x,y,z):
        
        a = 20
        b = 0.2
        c = 2*np.pi
        
        x = np.array([x,y,z])
    
#       d = -a*np.exp(-b*(np.sum(x**2, axis = 1)/x.shape[0])**0.5)
#       e = -np.exp(np.sum(np.cos(c*x), axis = 1)/x.shape[0])
    
        d = -a*np.exp(-b*(np.sum(x**2)/x.shape[0])**0.5)
        e = -np.exp(np.sum(np.cos(c*x)/x.shape[0]))
    
        return -(d+e+a+np.exp(1))


    #y = ackley(np.array([[3,2],[0,0]]))


    bopt = BayesOpt(ackley,
                    {'x' : (-32.768, 32.768),
                     'y' : (-32.768, 32.768),
                     'z' : (-32.768, 32.768)})       
        
    
    