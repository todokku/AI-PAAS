# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Wed Oct 23 13:44:59 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student
"""
import mlflow   as mf
import Training as tr

def cMAPPS(prepros_params,
           train_params,
           path = None,
           dataset_no):
    
    from Input      import cMAPSS as ci
    from Preprocess import cMAPSS as CP
    
    with mf.start_run():
    
        if path != None:
            ci.set_datapath(path)
        
        if dataset_no in range(1,5):
            ci.get_data(dataset_no)
            mf.log_param("DataSet Number", dataset_no)
        else:
            raise Exception('Please choose a number between 1 and 4')
             
        cp = CP(**prepros_params)
        cp.train_preprocess(ci.Train_input)

        lstm_ff = tr.LSTM_to_FF(cp.features,
                                run_id,
                                **train_params)
        lstm_ff.create_model()
        lstm_ff.train_model(train_in = cp.train_in, train_out = cp.train_out)
        
        mf.log_param('Features' , cp.features)
        mf.log_params(prepos_params)
        
        del train_params['enable_checkp']
        
        mf.log_params(train_params)
        mf.log_metrics({'MSE_Train'      : lstm_ff.loss,
                        'MSE_Validation' : lstm_ff.val_loss})
        
        #Tags
    