import math
import numpy as np
import math
from sklearn.metrics import mean_squared_error

def rul_health_score(y_true, y_pred):
    s=0

    d = y_pred - y_true

    #print(d)

    for i in range(len(d)):
        if d[i] < 0:
            s+=math.e**(-d[i]/13)-1
        else:
            s+=math.e**(d[i]/10)-1

    s = s/len(d)

    return s

def root_mean_squared_error(y_true, y_pred):
    
    s = math.sqrt(mean_squared_error(y_true, y_pred))

    return s

def accuracy(y_true, y_pred):

    argmax_true = np.argmax(y_true, axis=1)
    argmax_false = np.argmax(y_pred, axis=1)

    aux = argmax_true - argmax_false
    accurate = [1 if e==0 else 0 for e in aux]

    accuracy = np.mean(accurate)

    return accuracy

def epsilon_rms(y_pred, y_real, deltas):
    y_error = y_pred - y_real
    numerator = np.dot(y_error, y_error.transpose()) * deltas ** 2
    denominator = np.multiply(np.sum(y_real), deltas ** 2)

    epsilon = np.sqrt(np.asscalar(numerator / denominator))

    return epsilon

score_dict = {'rhs':lambda x,y : rul_health_score(x,y), 'rmse':lambda x,y : root_mean_squared_error(x,y),
              'mse':lambda x,y : mean_squared_error(x,y), 'accuracy':lambda x, y : accuracy(x, y),
              'epsilon_rms':lambda x,y,deltas : epsilon_rms(x,y,deltas)}


def compute_score(score_name, y_true, y_pred):
    
    score = score_dict[score_name](y_true, y_pred)
    return score


