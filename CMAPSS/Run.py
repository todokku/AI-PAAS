# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Wed Oct 23 13:44:59 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student
"""
import mlflow       as mf


#import mlflow.entities as run_md

def cMAPPS(prepros_params,
           train_params,
           dataset_no,
           path = None):
    
    from   Input      import cMAPSS as ci
    from   Preprocess import cMAPSS as CP
    from   Testing    import cMAPSS as ct 
    import Training                 as tr
    
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
        
        run_id = mf.active_run().info.run_id

        lstm_ff = tr.LSTM_to_FF(cp.features,
                                **train_params,
                                run_id = run_id)
        lstm_ff.create_model()
        
        lstm_ff.train_model(cp.train_in, cp.train_out)
        
        mf.log_param('Features' , cp.features)
        mf.log_params(prepros_params)
        
        del train_params['enable_checkp']
        
        mf.log_params(train_params)
        mf.log_metrics({'MSE_Train'      : lstm_ff.loss,
                        'MSE_Validation' : lstm_ff.val_loss,
                        'Delta_MSE'      : abs(lstm_ff.loss-lstm_ff.val_loss)})
    
        path = '../TrackedModels/' + run_id + '.hdf5'
    
        mf.log_artifact(path)
        
                
        #Tags
        
        mf.set_tags({'RMSE_Train'      : lstm_ff.loss**0.5,
                     'RMSE_Validation' : lstm_ff.val_loss**0.5})
       
        cp.test_preprocess(ci.Test_input)
        ct.get_score(lstm_ff.model, cp.test_in, ci.RUL_input)
        
        mf.log_metric({'Score'     : ct.s,
                       'Test_RMSE' : ct.rmse})

     