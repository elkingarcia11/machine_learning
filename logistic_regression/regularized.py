import numpy as np
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid
np.set_printoptions(precision=8)

def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """
    # m training examples
    # n features
    # first value of X.shape is training examples (rows)
    # second value of X.shape is features (columns)
    m,n  = X.shape

    # Track cost
    cost = 0.

    # for each training example
    for i in range(m):
        # get Z_i which is w_i*x_i + b
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        # get predicted value of nodel for that x,w,b
        f_wb_i = sigmoid(z_i)                                          #scalar
        # add cost of example to total cost
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar

    # divide total cost by training examples
    cost = cost/m                                                      #scalar
    # track regularization cost
    reg_cost = 0
    # for each feature
    for j in range(n):
        # add weight squared of that feature to reg cost 
        reg_cost += (w[j]**2)                                          #scalar
    # multiply total reg cost by lambda/2*m
    # where lambda is regularization weight 
    # and m is training examples
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar

def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """
    # Training examples
    m  = X.shape[0]

    # Features in model
    n  = len(w)

    # Total cost
    cost = 0.

    # For each training example
    for i in range(m):
        # Get predicted value
        f_wb_i = np.dot(X[i], w) + b                                   #(n,)(n,)=scalar, see np.dot

        # Get squared difference and add to cost
        cost = cost + (f_wb_i - y[i])**2                               #scalar  
    # Divide total cost by 2 * training examples           
    cost = cost / (2 * m)                                              #scalar  
    
    # Regularized cost
    reg_cost = 0

    # for each feature 
    for j in range(n):
        # add weight squared to reg cost
        reg_cost += (w[j]**2)                                          #scalar
    # multiply total reg cost by lambda/2*m
    # where lambda is weight of regularization and m is training examples
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    # total cost is mean squared error cost + regularization cost
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar

def compute_gradient_linear_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """

    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))  # 1D array of zeros with length n
    dj_db = 0.              # initial change in bias

    # for each training example
    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]          # calculate diff between predicted value and target value          
        for j in range(n):                          # for each feature
            dj_dw[j] = dj_dw[j] + err * X[i, j]     # make change in weight for this feature = current change in weight + diff * value of parameter
        dj_db = dj_db + err                         # make change in bias = current change in bias + diff
    dj_dw = dj_dw / m                               # divide change in weight by # of training examples
    dj_db = dj_db / m                               # divide change in bias by # of training examples
    
    # for each feature
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]    # change in weight for this feature = current change in weaight for this feature + regularization weight which is current weight * lambda/m

    return dj_db, dj_dw

def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape                                     # (training examples, features)
    dj_dw = np.zeros((n,))                            # array of 0s with length of features
    dj_db = 0.0                                       # scalar
    
    for i in range(m):                                # for each training example
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          # calculate predicted value given x, w, b
        err_i  = f_wb_i  - y[i]                       # calclate difference between predicted value and target value
        for j in range(n):                            # for each feature
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      # calculate change of feature weight by adding current feature weight change + diff * value of x
        dj_db = dj_db + err_i                         # calculate change of bias by adding current change of bias + difference
    dj_dw = dj_dw/m                                   # divide all change of weight by training examples
    dj_db = dj_db/m                                   # divide change in bias by training examples

    for j in range(n):                                # for each feature
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]      # add the current weight * lambda/m to the change in weight for that feature

    return dj_db, dj_dw

# For consistent random numbers
np.random.seed(1)

# Training data of 5 row each with 5 columns with values between 0 and 1
X_tmp = np.random.rand(5,6)
# Training target values
y_tmp = np.array([0,1,0,1,0])
# random weight assigned to same shape as x_temp, reshape to 1d array [-1] which is 5 items and values shifted down by .5 [-0.5]
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
# initial bias
b_tmp = 0.5
# initial lambda value
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

# Consistent random numbers
np.random.seed(1)
# Random training data with 5 examples and 5 features each
X_tmp = np.random.rand(5,6)
# Training target values
y_tmp = np.array([0,1,0,1,0])
# Random weights shifted down 0.5 and reshape to be the size total training examples
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
# initial bias
b_tmp = 0.5
# initial lambda
lambda_tmp = 0.7
# total cost for log reg
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )