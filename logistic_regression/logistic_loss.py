import numpy as np
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('./deeplearning.mplstyle')

# Squared error cost in linear regression
soup_bowl()

# Training data
x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)

# Logistic squared error cost vs (w,b)
plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()

# Loss curves for two categorical values where y = 1 and where y = 0
plt_two_logistic_loss_curves()

# Logistic cost graph vs (w,b ) and log(logistic curve) vs w,b
plt.close('all')
cst = plt_logistic_cost(x_train,y_train)