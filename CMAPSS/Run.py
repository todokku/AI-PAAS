"""
AI-AS ,Ryerson Univesity

@author:
    Tejas Janardhan
    AI-AS Phd Student

"""
import tempfile
import mlflow
from EstimatorCMAPSS import Estimator

params = {'ds_no': 3,
          'out_seq': True,
          'enable_dimred': True,

          'window_len': 41,
          'poly_order': 3,
          'var_threshold': 0.9,
          'conf_factor': 0.,

          'rnn_neurons': [30, 30],
          'ff_neurons': [20, 20],
          'rnn_type': 'LSTM',

          'epochs': 20,

          'lRELU_alpha': 0.3,
          'lr': 0.002,
          'dropout': 0.4,
          'rec_dropout': 0.2,
          'l2_k': 0.1,
          'l2_b': 0.,
          'l2_r': 0.,
          'enable_norm': False,
          'final_activation': 'softplus'}

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
