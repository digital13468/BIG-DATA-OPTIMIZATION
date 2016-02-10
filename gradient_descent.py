import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression 
import pylab
from scipy import stats

def gradient_descent(alpha, A, b, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    #m = x.shape[0] # number of samples
    #m, n = np.shape(x)
    # initial theta
    #t0 = np.random.random(x.shape[1])
    #t1 = np.random.random(x.shape[1])
    x = np.random.random(A.shape[1])
    #A_transpose = x.transpose()
    # total error, J(theta)
    #J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)]) / (2 * m)
    old_obj = np.sum((np.dot(A, x) - b) ** 2) / (2 * A.shape[0])
    print 'iter %s | objective: %.9f' % (iter, old_obj)
    # Iterate Loop
    while not converged:
        iter += 1  # update iter
        # for each training sample, compute the gradient (d/d_theta j(theta))
        #grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) 
        #grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])
        grad = np.dot(A.transpose(), (np.dot(A, x) - b)) / A.shape[0]
        # update the theta_temp
        #temp0 = t0 - alpha * grad0
        #temp1 = t1 - alpha * grad1
        
        # update theta
        #t0 = temp0
        #t1 = temp1
        x = x - alpha * grad
        new_obj = np.sum((np.dot(A, x) - b) ** 2) / (2 * A.shape[0])   # cost
        # mean squared error
        #e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] ) / (2 * m)
        print 'iter %s | objective: %.9f | gradient: %s' % (iter, new_obj, grad)
        #if grad0 == 0 and grad1 == 0:
        #    converged = True
        #if abs(new_obj-old_obj) <= ep:
        #    print 'Converged, iterations: ', iter, '!!!'
        #    converged = True
        if iter == max_iter:
            print 'Max interactions exceeded!'
            converged = True
        #J = e   # update error 
        #old_obj = new_obj
    
        

    return x

if __name__ == '__main__':

    A, b = make_regression(n_samples=100, n_features=1, n_informative=1, 
                        random_state=0, noise=35)
    A = np.c_[ A, np.ones(A.shape[0])]
    print 'A.shape = %s b.shape = %s' %(A.shape, b.shape)
    #m, n = np.shape(A)
    #m = A.shape(0)
    alpha = 0.01 # learning rate
    ep = 0.00001 # convergence criteria

    # call gredient decent, and get intercept(=theta0) and slope(=theta1)
    #theta0, theta1 = gradient_descent(alpha, x, y, ep, max_iter=1000000)
    #x = np.c_[ np.ones(m), x] # insert column
   
    x = gradient_descent(alpha, A, b, ep, max_iter=1000)
    print ('x = %s') %(x) 
    
    # check with scipy linear regression 
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(A[:,0], b)
    e = sum( [ (intercept*A[i,1] + slope*A[i,0] - b[i])**2 for i in range(A.shape[0])] ) / (2*A.shape[0])
    print ('intercept = %s slope = %s objective = %s') %(intercept, slope, e)
    #print ('%s') %(np.dot(x.transpose, x))
    algSol = np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose()), b)
    #algObj = sum( [ (algSol[0] + algSol[1]*x[i,1] - y[i])**2 for i in range(x.shape[0])] ) / (2*x.shape[0])
    #algObj = sum((np.dot(A[i,:], algSol) - b[i])**2 for i in range(A.shape[0])) / (2*A.shape[0])
    algObj = np.sum((np.dot(A, algSol) - b) ** 2) / (2 * A.shape[0])
    print ('algSol = %s algObj = %s') %(algSol, algObj)
    # plot
    #for i in range(x.shape[1]):
    b_predict = np.dot(A, x)#theta[0] + theta[1]*x 

    pylab.plot(A[:,0],b,'o')
    pylab.plot(A[:,0],b_predict,'k-')
    pylab.show()
    print "Done!"
