
import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression 
import pylab
from scipy import stats
import time
import datetime
import os
from io import StringIO

def plot_distribution(digit, data):
    pylab.figure('Digits')
    pylab.subplot(2, 5, digit+1)
    b = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        if data[i,0] == digit:
            b[i] = 1
            pylab.plot(data[i,1], data[i,2], 'b*')
        else:
            b[i] = -1
            pylab.plot(data[i,1], data[i,2], 'r.')

    pylab.title('digit-' + str(digit))
    pylab.xlabel('Intensity')
    pylab.ylabel('Symmetry')
    A = np.c_[np.delete(data, 0, 1), np.ones(data.shape[0])]
    return A, b
    
def contour_image(A, b, x):
    lx = -8.0
    ly = -8.0
    rx = 8.0
    ry = 8.0
    
    
    num = 9
    
    x1 = np.linspace(lx, rx, num)
    x2 = np.linspace(ly, ry, num)
    (X1, X2) = np.meshgrid(x1, x2)
    
    z = [[0 for i in range(x1.size)] for j in range(x2.size)]
    for i in range(x1.size):
        for j in range(x2.size):
            z[i][j] = sum([(x1[i]*A[k,0] + x2[j]*A[k,1] - b[k])**2 for k in range(A.shape[0])]) / (2)
    arr_x = np.asarray(x)

    CS = pylab.contour(x1, x2, z, colors='k')
    pylab.clabel(CS, inline=1, fontsize=10) 
    pylab.plot(arr_x[:,0],arr_x[:,1])
    pylab.axis([lx, rx, ly, ry])
    
def function_value(A, x, b):
    return np.sum((np.dot(A, x) - b) ** 2) / (2)
    
def armijo_rule(s, beta, sigma, A, x, b, grad):
    alpha = s
    while function_value(A, x, b) - function_value(A, x-(alpha*grad), b) < sigma*alpha*(np.dot(grad.transpose(), grad)):
        alpha = beta * alpha 
    return alpha
    
def gradient_descent(rule, A, b, obj_list, x, x_list, max_iter=10000, alpha = 0, s = 0, beta = 0, sigma = 0):
    converged = False
    iter = 0
    
    x_list.append(x)
    
    old_obj = np.sum((np.dot(A, x) - b) ** 2) / (2)
    print 'iter %s | objective: %.3f' % (iter, old_obj)
    obj_list.append(old_obj)
    
    while not converged:
        iter += 1  
        
        grad = np.dot(A.transpose(), (np.dot(A, x) - b))
        if rule == 'constant':
            alpha = alpha
            
        elif rule == 'diminishing-1':
            alpha = float(0.01) / iter
            
        elif rule == 'diminishing-2':
            alpha = float(0.01) / ((iter)**2)
            
        elif rule == 'diminishing-1/2':
            alpha = (0.01/(iter)**0.5)
            
        elif rule == 'armijo':
            alpha = armijo_rule(s, beta, sigma, A, x, b, grad)
        else:
            print "Got a false value"
        x = x - alpha * grad
        x_list.append(x.tolist())
        
        
        new_obj = np.sum((np.dot(A, x) - b) ** 2) / (2)   
        obj_list.append(new_obj)
        strings = ["%e" % number for number in x]
        print 'iter %s | objective: %e | x: %s | alpha: %e' % (iter, new_obj, strings, alpha)
        if obj_list[-1] == obj_list[-2] and obj_list[-2] == obj_list[-3]:
            converged = True
        
        if iter == max_iter:
            print 'Max interactions exceeded!'
            converged = True
        
    
        

    return x

def init(rule, A, x, b, max_iter, fig, alpha = 0, s = 0, beta = 0, sigma = 0):
    obj_list = []
    x_list = []
    x = gradient_descent(rule, A, b, obj_list, x, x_list, max_iter, alpha, s, beta, sigma)
    #print ('x = %s') %(x_list) 
    #print ('obj = %s') %(obj_list)
    #b_predict = np.dot(A, x)
    #pylab.figure()
    #pylab.subplot(3, 1, 1)
    #fig[0] = fig[0] + 1
    #pylab.plot(A[:,0],b,'o')
    #pylab.plot(A[:,0],b_predict,'k-')
    #if (rule == 'constant'):
    #    pylab.title(rule + '-' + str(alpha))
    #elif (rule == 'armijo'):
    #    pylab.title(rule + '(s/beta/sigma) - ' + str(s) + '/' + str(beta) + '/' + str(sigma))
    #else:
    #    pylab.title(rule)
    
    iter_list = []
    for i in range(len(obj_list)):
        iter_list.append(i)
    pylab.subplot(7, 1, fig[0])
    #pylab.subplot(3, 1, 2)

    fig[0] = fig[0] + 1
    if obj_list[-1] == np.inf:
        pylab.annotate('('+str(iter_list[-1])+', diverged)', xy=(iter_list[-4], obj_list[-4]), xytext=(iter_list[-4]*0.8, obj_list[-4]*0.8), arrowprops=dict(facecolor='black', shrink=0.03))
    else:
        pylab.annotate('('+str(iter_list[-1])+', '+str(round(obj_list[-1], 3))+')', xy=(iter_list[-1], obj_list[-1]), xytext=(iter_list[-1]*0.8, obj_list[1]), arrowprops=dict(facecolor='black', shrink=0.03))
    pylab.plot(iter_list, obj_list, lw=3)
    pylab.xlabel('Number of Iteration')
    pylab.ylabel('Objective Value')
    if (rule == 'constant'):
        pylab.title(rule + '-' +str(round(alpha, 6)))
    elif (rule == 'armijo'):
        pylab.title(rule + '(s/beta/sigma) - ' + str(round(s, 3)) + '/' + str(round(beta, 3)) + '/' +str(round(sigma, 3)))
    else:
        pylab.title(rule)
    pylab.xlim([0, iter_list[-1]])
    #pylab.subplot(3, 1, 3)
    #contour_image(A, b, x_list)
    
    
def stepsize_rules(rule, A, x, b, max_iter, fig, s = 0, beta = 0, sigma = 0):
    ts = time.time()
    if rule == 'constant':
        alpha = np.random.random() * 0.001
        print '   =========================   %s - %e   =========================' % (rule, alpha)
        print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        init(rule, A, x, b, max_iter, fig, alpha)
    
        alpha = np.random.random() * 0.0001
        print '   =========================   %s - %e   =========================' % (rule, alpha)
        print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        init(rule, A, x, b, max_iter, fig, alpha)
    
    elif rule == 'diminishing-1':
        print ('   ============================   ' + rule + '   ============================')
        print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        init(rule, A, x, b, max_iter, fig)
    
    elif rule == 'diminishing-2':
        print ('   ============================   ' + rule + '   =============================')
        print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        init(rule, A, x, b, max_iter, fig)
    
    elif rule == 'diminishing-1/2':
        print ('   ============================   ' + rule + '   ============================')
        print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        init(rule, A, x, b, max_iter, fig)
    
    elif rule == 'armijo':
        print ('   ======================   ' + rule + '(s/beta/sigma) - ' + str(round(s, 3)) + '/' + str(round(beta, 3)) + '/' + str(round(sigma, 3)) + '   ======================')
        print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        init(rule, A, x, b, max_iter, fig, 0, s, beta, sigma)
    
    else:
        print "Got a false value"
    

def data_input(interval, row, column, data_type):
    if data_type == 'matrix':
        m = np.random.random_integers(interval, size=(row,column))
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i][j] = m[i][j] * random.uniform(-1, 1)
    elif data_type == 'vector':
        m = np.random.random_integers(interval, size=row)
        for i in range(m.shape[0]):
            m[i] = m[i] * random.uniform(-1, 1)
    else:
        print "Got a false value"
    return m

def running_algorithm(A, b, max_iter):
        
    print 'A.shape = %s b.shape = %s' %(A.shape, b.shape)
    print 'A = %s' %(A)
    print 'b = %s' %(b)
    fig = [1]
    x = np.random.random(A.shape[1]).tolist()
    
    stepsize_rules('constant', A, x, b, max_iter, fig)
    stepsize_rules('diminishing-1', A, x, b, max_iter, fig)
    stepsize_rules('diminishing-2', A, x, b, max_iter, fig)
    stepsize_rules('diminishing-1/2', A, x, b, max_iter, fig)
    stepsize_rules('armijo', A, x, b, max_iter, fig, s = random.uniform(0.1, 1.1), beta = random.uniform(0.1, 1), sigma = random.uniform(0.1, 0.5))
    stepsize_rules('armijo', A, x, b, max_iter, fig, s = random.uniform(0.1, 1.1), beta = random.uniform(0.1, 1), sigma = random.uniform(0.1, 0.5))
    print ('   ============================   optimality   ============================')
 
    algSol = np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose()), b)
    
    algObj = np.sum((np.dot(A, algSol) - b) ** 2) / (2)
    print ('algSol = %s\nalgObj = %s') %(algSol, algObj)

    
    
    w, v = np.linalg.eig(np.dot(A.transpose(), A)) 
    sort_w = sorted(w)
    print 'the eigenvalues: %s' %(w)
    print 'the condition number: %s' %(1-sort_w[0]/sort_w[len(sort_w)-1])
    #pylab.show()

def data_set_problem(directory, filename, max_iter):
    f = open(os.path.join(directory, filename), 'r')
    data = np.genfromtxt(directory+'/'+filename)
    f.close()
    A_digit = {}
    
    for i in range(10):   
        A_digit[i] = plot_distribution(i, data)
    #pylab.show()
    for i in range(10):        
        A = A_digit[i][0]
        b = A_digit[i][1]
        pylab.figure('digit-'+str(i))
        running_algorithm(A, b, max_iter)

def data_set_problem1(directory, filename, max_iter, sep_a, sep_b):
    if sep_a < 0 or sep_a > 9 or sep_b < 0 or sep_b > 9:
        print 'not digit number'
    f = open(os.path.join(directory, filename), 'r')
    data = np.genfromtxt(directory+'/'+filename)
    f.close()
    A_digit = {}
    pylab.figure('digit-'+str(sep_a)+'/'+str(sep_b))
    #for i in range(10):   
    #    A_digit[i] = plot_distribution(i, data)
    #pylab.show()
    
    b = np.empty((0,))
    A = np.empty((0,0))
    for i in range(data.shape[0]):
        if data[i,0] == sep_a:
            #print '%s' %(np.ndim(A))
            A = np.append(A, data[i,:])
            A = A.reshape((len(A)/data[i,:].size,data[i,:].size))
            b = np.append(b, 1)
            pylab.plot(data[i,1], data[i,2], 'b*')
        elif data[i,0] == sep_b:
            A = np.append(A, data[i,:])
            A = A.reshape((len(A)/data[i,:].size,data[i,:].size))
            b = np.append(b, -1)
            pylab.plot(data[i,1], data[i,2], 'r.')
        
    A = np.c_[np.delete(A, 0, 1), np.ones(A.shape[0])]
    pylab.title('digit-'+str(sep_a)+'/'+str(sep_b))
    pylab.xlabel('Intensity')
    pylab.ylabel('Symmetry')
    
    pylab.figure('Objective Reduction when Seperating Digit '+str(sep_a)+' with '+str(sep_b))
    running_algorithm(A, b, max_iter)

def randomly_generated_problem(max_iter):
    #A, b = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35)
    #A = np.c_[ A, np.ones(A.shape[0])]
    #A = np.random.rand(50, 10)
    #b = np.random.rand(50)
    #A = np.random.random_integers(3, size=(50.,2.))
    #b = np.random.random_integers(3, size=50)
    A = data_input(8, 50, 10, 'matrix')
    b = data_input(8, 50, 1, 'vector')
    pylab.figure('randomly generated problem')
    running_algorithm(A, b, max_iter)
    
if __name__ == '__main__':
    #ep = 0.00001 # convergence criteria
    max_iter = 2000
    sep_a = 6
    sep_b = 7
    data_set_problem('handwriting data', 'features_train.txt', max_iter)
    randomly_generated_problem(max_iter)
    data_set_problem1('handwriting data', 'features_train.txt', max_iter, sep_a, sep_b)
    pylab.show()
    
    
    
    
    print "Done!"
