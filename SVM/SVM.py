from plot import *
from SVM_dual import *
from BCGD import *
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
from perceptron import *
from LRmodel import *
from linear_support_vector_machine import *

def data_preprocess(digit, data):
    #pylab.figure('Digits')
    #pylab.subplot(2, 5, digit+1)
    b = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        if data[i,0] == digit:
            b[i] = 1
            #pylab.plot(data[i,1], data[i,2], 'b*')
        else:
            b[i] = -1
            #pylab.plot(data[i,1], data[i,2], 'r.')

    #pylab.title('digit-' + str(digit))
    #pylab.xlabel('Intensity')
    #pylab.ylabel('Symmetry')
    A = np.c_[np.delete(data, 0, 1), np.ones(data.shape[0])]
    return A, b
    
def test_data_preprocess(digit, test_data):
    
    b = np.zeros(test_data.shape[0])
    
    for i in range(test_data.shape[0]):
        if test_data[i,0] == digit:
            b[i] = 1
        else:
            b[i] = -1
            
    A = np.c_[np.delete(test_data, 0, 1), np.ones(test_data.shape[0])]
    
    return A, b
    
def function_value(A, x, b):
    return np.sum((np.dot(A, x) - b) ** 2) / (2)
    
def armijo_rule(s, beta, sigma, A, x, b, grad):
    alpha = s
    while function_value(A, x, b) - function_value(A, x-(alpha*grad), b) < sigma*alpha*(np.dot(grad.transpose(), grad)):
        alpha = beta * alpha 
    return alpha
    
def gradient_descent(A_test, b_test, color, fig_error, fig_line, rule, A, b, obj_list, x_init, x_list, max_iter=10000, alpha = 0, s = 0, beta = 0, sigma = 0):
    converged = False
    iter = 0
    x = x_init[:]
    x_list.append(x)
    #print x
    old_obj = np.sum((np.dot(A, x) - b) ** 2) / (2)
    time_diff_list = []
    obj_list.append(old_obj)
    iter_list = create_iter_list(max_iter)
    tstart = datetime.datetime.now()
    training_error_list = []
    testing_error_list = []
    time_diff_list.append(0)
    print 'iteration 0 at ',
    print tstart
    training_error_list.append(error_count(x_list[-1], A, b, 'training'))
    testing_error_list.append(error_count(x_list[-1], A_test, b_test, 'testing'))
    while not converged:
        iter += 1  
        
        grad = np.dot(A.transpose(), (np.dot(A, x) - b))
        if rule == 'constant':
            alpha = alpha
            
        elif rule == 'diminishing-1/r':
            alpha = float(0.01) / iter
            
        elif rule == 'diminishing-1/r^2':
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
        #print 'iter %s | objective: %e | x: %s | alpha: %e' % (iter, new_obj, strings, alpha)
        #if obj_list[-1] == obj_list[-2] and obj_list[-2] == obj_list[
        #    converged = True
        
        if iter == max_iter:
            #print 'Max interactions exceeded!'
            converged = True
        if iter in iter_list:
            tend = datetime.datetime.now()
            time_diff_list.append((tend-tstart).total_seconds()*1000)
            print 'iteration %s time ' %(iter),
            print tend
            training_error_list.append(error_count(x_list[-1], A, b, 'training'))
            testing_error_list.append(LRerror(x_list[-1], A_test, b_test, 'testing'))
            #tstart = tend
        if iter == 1 or iter == 5 or iter == 10 or iter == 50:
            plot_classifier(A, x, rule, fig_line, alpha, s, beta, sigma)
    plot_error(color, fig_error, rule, iter_list, training_error_list, testing_error_list)
    return x

def init(A_test, b_test, rule, color, fig_error, A, x_init, b, max_iter, fig_objective, fig_line, alpha = 0, s = 0, beta = 0, sigma = 0):
    obj_list = []
    x_list = []
    x_converge = gradient_descent(A_test, b_test, color, fig_error, fig_line, rule, A, b, obj_list, x_init, x_list, max_iter, alpha, s, beta, sigma)
    
    plot_objective_evolution(fig_objective, obj_list, rule, alpha, s, beta, sigma)
    #plot_classifier(A, x_converge, rule, fig_line, alpha, s, beta, sigma)
           
    return x_list, obj_list
    
    
def stepsize_rules(rule, color, fig_error, A, x, b, max_iter, fig_objective, fig_line, lambda_max, A_test, b_test, s = 0, beta = 0, sigma = 0):
    ts = time.time()
    if rule == 'constant':
        #alpha = np.random.random() * 0.001
        #print '   ======================   %s - %e   ======================' % (rule, alpha)
        ##print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        #x_list, obj_list = init(rule, A, x, b, max_iter, fig_objective, fig_line, alpha)
        #error_count(x_list[-1], A, b, 'training')
        #error_count(x_list[-1], A_test, b_test, 'testing')
        ##print x_list[-1],
        #print obj_list[-1]
        #print
        
        alpha = lambda_max
        print '   ======================   %s - %e   ======================' % (rule, alpha)
        #print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        x_list, obj_list = init(A_test, b_test, rule, color, fig_error, A, x, b, max_iter, fig_objective, fig_line, alpha)
        #error_count(x_list[-1], A, b, 'training')
        #error_count(x_list[-1], A_test, b_test, 'testing')
        #print x_list[-1],
        print obj_list[-1], len(obj_list)
        print
        
    elif rule == 'diminishing-1/r':
        print ('   =========================   ' + rule + '   =========================')
        #print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        x_list, obj_list = init(rule, A, x, b, max_iter, fig_objective, fig_line)
        error_count(x_list[-1], A, b, 'training')
        error_count(x_list[-1], A_test, b_test, 'testing')
        #print x_list[-1],
        print obj_list[-1], len(obj_list)
        print
        
    elif rule == 'diminishing-1/r^2':
        print ('   =========================   ' + rule + '   ==========================')
        #print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        x_list, obj_list = init(rule, A, x, b, max_iter, fig_objective, fig_line)
        error_count(x_list[-1], A, b, 'training')
        error_count(x_list[-1], A_test, b_test, 'testing')
        #print x_list[-1],
        print obj_list[-1], len(obj_list)
        print
    
    elif rule == 'diminishing-1/2':
        print ('   =========================   ' + rule + '   =========================')
        #print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        x_list, obj_list = init(rule, A, x, b, max_iter, fig_objective, fig_line)
        error_count(x_list[-1], A, b, 'training')
        error_count(x_list[-1], A_test, b_test, 'testing')
        print x_list[-1], obj_list[-1], len(obj_list)
        ptint
    
    elif rule == 'armijo':
        print ('   =============   ' + rule + '(s/beta/sigma) - ' + str(round(s, 3)) + '/' + str(round(beta, 3)) + '/' + str(round(sigma, 3)) + '   =============')
        #print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        x_list, obj_list = init(A_test, b_test, rule, color, fig_error, A, x, b, max_iter, fig_objective, fig_line, 0, s, beta, sigma)
        #error_count(x_list[-1], A, b, 'training')
        #error_count(x_list[-1], A_test, b_test, 'testing')
        #print x_list[-1]
        print obj_list[-1], len(obj_list)
        print
    
    else:
        print "Got a false value"
        sys.exit()
    

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
        sys.exit()
    return m

def running_algorithm(color, fig_error, A, b, max_iter, x, fig_objective, fig_line, A_test, b_test):
    #classifier_matrix = []  
    #print 'A.shape = %s b.shape = %s' %(A.shape, b.shape)
    #print 'A = %s' %(A)
    #print 'b = %s' %(b)
    #fig = [1]
    #x = np.random.random(A.shape[1]).tolist()
    algSol = np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose()), b)
    w, v = np.linalg.eig(np.dot(A.transpose(), A)) 
    sort_w = sorted(w)
    
    #plot_classifier(A, b, algSol, 'opt')
    print 'Solving linear regression model with different algorithms...'
    stepsize_rules('constant', color, fig_error, A, x, b, max_iter, fig_objective, fig_line, 1.0/sort_w[len(sort_w)-1], A_test, b_test)
    #stepsize_rules('diminishing-1/r', A, x, b, max_iter, fig_objective, fig_line, 1.0/sort_w[len(sort_w)-1], A_test, b_test)
    #stepsize_rules('diminishing-1/r^2', A, x, b, max_iter, fig_objective, fig_line, 1.0/sort_w[len(sort_w)-1], A_test, b_test)
    #stepsize_rules('diminishing-1/2', A, x, b, max_iter, fig_objective, fig_line, 1.0/sort_w[len(sort_w)-1], A_test, b_test)
    #stepsize_rules('armijo', A, x, b, max_iter, fig_objective, fig_line, 1.0/sort_w[len(sort_w)-1], A_test, b_test, s = random.uniform(0.1, 1.1), beta = random.uniform(0.1, 1), sigma = random.uniform(0.1, 0.5))
    stepsize_rules('armijo', color, fig_error, A, x, b, max_iter, fig_objective, fig_line, 1.0/sort_w[len(sort_w)-1], A_test, b_test, s = random.uniform(0.1, 1.1), beta = random.uniform(0.1, 1), sigma = random.uniform(0.1, 0.5))
    
    print ('   =========================   optimality   =========================') 
    algObj = np.sum((np.dot(A, algSol) - b) ** 2) / (2)
    print ('algObj = %s') %(algObj) 
    #print 'the eigenvalues: %s' %(w)
    #print 'the condition number: %s' %(1-sort_w[0]/sort_w[len(sort_w)-1])
    print
    
    #return classifier_matrix

def read_data(directory, filename):
    f = open(os.path.join(directory, filename), 'r')
    data = np.genfromtxt(directory+'/'+filename)
    f.close()
    return data

def data_set_problem(color, data, max_iter, x_init, test_data, data_type):
    visual_classifier = []
    A_digit = {}
    test_data_digit = {}
    #print test_data
    for i in range(10):   
        A_digit[i] = data_preprocess(i, data)
        test_data_digit[i] = test_data_preprocess(i, test_data)
     #   print test_data_digit[i]
    for i in range(1, 2):
        print 'Starting the task of separating digit %s from the rest on %s data...' %(str(i), data_type)
        
        A = A_digit[i][0]
        b = A_digit[i][1]
        A_test = test_data_digit[i][0]
        b_test = test_data_digit[i][1]
        
        fig_objective = 'Evolution of the objective function for digit-'+str(i)+' on %s data (linear regression)' %(data_type)
        fig_line = 'Line of classification for digit-'+str(i)+' on '+data_type+' data'
        fig_error = 'Error rate of classification for digit-'+str(i)+' on '+data_type+' data'
        create_objective_plot(fig_objective)
        create_line_plot(fig_line, A, b)
        create_error_plot(fig_error)
        running_algorithm(color, fig_error, A, b, max_iter, x_init, fig_objective, fig_line, A_test, b_test)
        #if data_type == 'featured':
        #    PBCGD(len(x_init), A, b, A_test, b_test, max_iter, x_init, fig_objective, fig_line)
        #    
        #    CBCGD(1, color, fig_error, A, b, A_test, b_test, max_iter, x_init, fig_objective, fig_line)
        #    RBCGD(2, color, fig_error, A, b, A_test, b_test, max_iter, x_init, fig_objective, fig_line)
        #else:
        #    PBCGD(153, A, b, A_test, b_test, max_iter, x_init, fig_objective, fig_line)
        #    CBCGD(153, A, b, A_test, b_test, max_iter, x_init, fig_objective, fig_line)
        #    RBCGD(153, A, b, A_test, b_test, max_iter, x_init, fig_objective, fig_line)
        print 'Linear regrssion on %s data done!' %(data_type)
        print
        margin_SVM(color, A, b, max_iter, i, fig_error, fig_line, 'Evolution of the objective function for digit-'+str(i)+' on '+data_type+' data', A_test, b_test)
        print 'Dual SVM on %s data done!' %(data_type)
        print
        #logistic_regression(color, fig_error, A, b, x_init, A_test, b_test, max_iter, i, fig_line, 'Evolution of the objective function for digit-'+str(i)+' on '+data_type+' data')
        #linear_SVM(color, A, b, x_init, max_iter, i, fig_error, fig_line, 'Evolution of the objective function for digit-'+str(i)+' on '+data_type+' data', A_test, b_test)
        #print 'Linear SVM on %s data done!' %(data_type)
        #PLA_problem_setup(color, A, b, x_init, max_iter, i, fig_error, fig_line, 'Evolution of the objective function for digit-'+str(i)+' on '+data_type+' data', A_test, b_test)
        #print 'PLA on %s data done!' %(data_type)
        print 'Plotting figures...'
        objective_plot_define(fig_objective, max_iter)#, ylim = 9000, xlim = -9)
        line_plot_define(fig_line)
        error_plot_define(fig_error)
       
        
def data_set_problem1(data, max_iter, sep_a, sep_b, x_init, test_data):
    if sep_a < 0 or sep_a > 9 or sep_b < 0 or sep_b > 9:
        print 'not digit number'
        sys.exit()
    A_digit = {}
    pylab.figure('digit-'+str(sep_a)+'/'+str(sep_b))
    
    
    b = np.empty((0,))
    A = np.empty((0,0))
    for i in range(data.shape[0]):
        if data[i,0] == sep_a:
            
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
    running_algorithm(A, b, max_iter, x_init)


    
if __name__ == '__main__':
    #ep = 0.00001 # convergence criteria
    max_iter = 50
    sep_a = 6
    sep_b = 7
    
    data = read_data('handwriting data', 'features_train.txt')
    test_data = read_data('handwriting data', 'features_test.txt')
    #raw_data = read_data('handwriting data', 'training_data_raw.txt')
    #raw_test_data = read_data('handwriting data', 'test_data_raw.txt')
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '']
    #colormap = pylab.cm.gist_ncar
    #color = [colormap(i) for i in range(14)]
    x_init = np.random.random(data.shape[1]).tolist()
    #x_init = np.zeros((data.shape[1],)).tolist()
    print 'The starting point on featured data is %s' %(x_init)
    data_set_problem(color, data, max_iter, x_init, test_data, 'featured')
    print 
    #x_init = np.random.random(raw_data.shape[1]).tolist()
    #print 'The starting point on raw data is %s' %(x_init)
    #data_set_problem(raw_data, max_iter, x_init, raw_test_data, 'raw')
    #data_set_problem1(data, max_iter, sep_a, sep_b, x_init, test_data)
    #print color
    pylab.show()
    
    
    
    
    print "Done!"
    
