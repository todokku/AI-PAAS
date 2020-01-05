# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Wed Oct 23 13:44:59 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student
"""

import mlflow
from tensorflow.keras.backend import clear_session
import tempfile

def cMAPPS(experiment_name,
           prepros_params,
           train_params,
           dataset_no,
           tracking = True,
           path     = None):
    
    from   Input      import cMAPSS as ci
    from   Preprocess import cMAPSS as CP
    from   Testing    import cMAPSS as ct 
    import Training                 as tr
    
    if tracking == True:
        
        mlflow.set_tracking_uri('sqlite:///mlflow.db')

        mlflow.set_experiment(experiment_name)
    
        with mlflow.start_run():
        
            if path != None:
                ci.set_datapath(path)
            
            if dataset_no in range(1,5):
                ci.get_data(dataset_no)
                mlflow.log_param("DataSet Number", dataset_no)
            else:
                raise Exception('Please choose a number between 1 and 4')
                 
            cp = CP(**prepros_params)
            cp.preprocess(ci.Train_input)
            
            run_id = mlflow.active_run().info.run_id
    
            tmpdir = tempfile.TemporaryDirectory()
            
            rnn_ff = tr.RNN_to_FF(cp.features,
                                  **train_params,
                                  model_dir = tmpdir.name,
                                  run_id    = run_id)
            rnn_ff.create_model(cp.no_splits)
            
            rnn_ff.train_model(cp.splits_in, cp.splits_out)
            
            mlflow.log_param('Features' , cp.features)
            mlflow.log_params(prepros_params)
    
            mlflow.log_params(train_params)
            mlflow.log_metrics({'MSE_Train'      : rnn_ff.loss,
                                'MSE_Validation' : rnn_ff.val_loss,
                                'Delta_MSE'      : abs(rnn_ff.loss-rnn_ff.val_loss)})
        
            mlflow.log_artifacts(tmpdir.name)
          
            tmpdir.cleanup()
            #Tags
            
            mlflow.set_tags({'RMSE_Train'      : rnn_ff.loss**0.5,
                             'RMSE_Validation' : rnn_ff.val_loss**0.5})
           
            cp.preprocess(ci.Test_input, isTrain = False)
            ct.get_score(rnn_ff.model, cp.test_in, ci.RUL_input)
            
            mlflow.log_metrics({'Score'    : ct.s,
                                'Test_MSE' : ct.mse})
        
            mlflow.set_tag('Test_RMSE', ct.mse**0.5)
            
    else:
        if path != None:
            ci.set_datapath(path)    
        if dataset_no in range(1,5):
            ci.get_data(dataset_no)
        else:
            raise Exception('Please choose a number between 1 and 4')
        cp = CP(**prepros_params)
        cp.preprocess(ci.Train_input)
        rnn_ff = tr.RNN_to_FF(cp.features,
                              **train_params)
        rnn_ff.create_model(cp.no_splits)
        rnn_ff.train_model(cp.splits_in, cp.splits_out, cp.no_splits)
        cp.preprocess(ci.Test_input, isTrain = False)
        ct.get_score(rnn_ff.model, cp.test_in, ci.RUL_input)
    
    clear_session()

    