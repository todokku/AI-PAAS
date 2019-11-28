# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Thu Nov 21 18:39:32 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student
    
"""

# =================================================================================================
# Create plots for individual signals or list of engines
# =================================================================================================

import seaborn

class cMAPSS:
    
    def __init__(self,
                 engine_no   = 1,
                 engine_list = 10,
                 engine_based = False): 
        
        self.engine_list  = engine_list
        self.engine_no    = engine_no
        self.engine_based = False
    
    
    
    def signal_plot(self):
        pass
    
    def engines_plot(self):
        pass