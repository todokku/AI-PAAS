import mlflow
import tempfile

mlflow.set_tracking_uri('sqlite:///mlflow.db')

with mlflow.start_run():
    tmpdir = tempfile.TemporaryDirectory()
    f = open(tmpdir.name + '\\Test.txt', 'w+')
    f.close()
    mlflow.log_param('Test1', [0, 1])

    mlflow.log_artifact(tmpdir.name + '\\Test.txt')
    tmpdir.cleanup()
