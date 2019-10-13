# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Candidate

"""

# =============================================================================
# NOT TO BE USED TO TWEAK THE HYPERPARAMETERS !!!!
# you can but only if you are zen
# =============================================================================


import numpy as np


class cMAPSS:
    
    @classmethod
    def get_score(cls, model, test_in, true_rul):
        
        cls.true_rul = true_rul.to_numpy()
        cls.est_rul = model.predict(test_in, batch_size=None)  
                
        #Calculating S score from the NASA paper, variables can be found there
        
        d       = cls.est_rul - cls.true_rul
        d[d<0]  = np.exp(-(d[d<0]/10)) - 1
        d[d>=0] = np.exp(d[d>=0]/13) - 1
        cls.s   = d.sum()
        
        print(f'The score is - {cls.s} !!! Cry or Celebrate')
        
        
        
        
        
        
        
        
        
        