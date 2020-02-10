"""
AI-AS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-AS Phd Student

"""
import tempfile
import mlflow
from EstimatorCMAPSS import Estimator

params = {'ds_no': 1,

          'window_len': 7,
          'poly_order': 3,
          'var_threshold': 0.7,
          'conf_factor': -0.25,
          's_len': 5,
          'initial_cutoff': 0.,
          'ins_dropped': 0.,

          'rnn_neurons': [10],
          'ff_neurons': [10],
          'rnn_type': 'LSTM',

          'epochs': 5,

          'lRELU_alpha': 0.3,
          'lr': 0.002,
          'dropout': 0.4,
          'rec_dropout': 0.2,
          'l2_k': 0.001,
          'l2_b': 0.,
          'l2_r': 0.,
          'enable_norm': False,
          'final_activation': 'None'}

mlflow.set_tracking_uri('sqlite:///mlflow.db')

with mlflow.start_run():
    tmpdir = tempfile.TemporaryDirectory()
    run_id = mlflow.active_run().info.run_id
    mlflow.log_params(params)

    estimator = Estimator(**params, run_id=run_id, model_dir=tmpdir.name)
    estimator.evaluate_params()

    mlflow.log_metrics({'Score': estimator.score,
                        'Training MSE': estimator.model_manager.mse,
                        'Training RMSE': estimator.model_manager.mse ** 0.5,
                        'Test MSE': estimator.tester.mse,
                        'Test RMSE': estimator.tester.mse ** 0.5})
