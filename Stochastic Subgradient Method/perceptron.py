import sys
import pylab
import numpy as np
from plot import *
import datetime

    
    

def generate_alpha(max_iter):
    
    constant1 = np.random.random() * 0.001
    constant2 = np.random.random() * 0.0001
    alpha = np.empty((4,max_iter))
    for i in range(alpha.shape[0]):
        for j in range(max_iter):
            if i == 0:
                alpha[i][j] = constant1
            elif i == 1:
                alpha[i][j] = constant2
            elif i ==3:
                alpha[i][j] = 1.0/i
            elif i == 4:
                alpha[i][j] = 1.0/(i*i)
    return alphas

def PLA_problem_setup(color, A, b, x_init, max_iter, digit, fig_error, fig_line, fig_title, A_test, b_test):
    print 'Solving PLA model with incremental algorithm...'
    alphas = [1]
    #pylab.figure('Modified PLA Method for Digit-'+str(digit))
    #pylab.xlabel('Number of Iteration')
    #pylab.ylabel('Objective Value')
    fig_objective = fig_title+' (PLA)'
    create_objective_plot(fig_objective)
    #classifier_matrix = []
    for i in range(len(alphas)):
        pocket_algorithm(color, b, A, x_init, max_iter, alphas[i], fig_error, fig_line, fig_objective, A_test, b_test)
    #pylab.legend()
    objective_plot_define(fig_objective, max_iter)#, ylim = 900, xlim = 0)
    
    #return classifier_matrix

#def error_rate(x, A, b):
#    error = 0.0
#    for i in range(A.shape[0]):
#        if b[i] * np.dot(A[i,:], x) < 0:
#            error += 1
#            
#    return error/A.shape[0]

def pocket_algorithm(color, b, A, x_init, max_iter, alpha, fig_error, fig_line, fig_objective, A_test, b_test):
    print ('   =========================   alpha = %s   =========================') %(str(alpha))
    x_list = []
    x_list.append(x_init)
    objective_value_history = []
    objective_value_history.append(evaluate_objective_value(b, A, x_init))
    time_diff_list = []
    x_iter = x_init[:]
    iter_list = create_iter_list(max_iter)
    training_error_list = []
    testing_error_list = []
    tstart = datetime.datetime.now()
    time_diff_list.append(0)
    print 'iteration 0 at ',
    print tstart
    training_error_list.append(error_count(x_list[-1], A, b, 'training'))
    testing_error_list.append(error_count(x_list[-1], A_test, b_test, 'testing'))
    for i in range(1, max_iter+1):
        prev_x = x_list[-1][:]
        #print x_iter
        for j in range(A.shape[0]):
            #print '%s %s' %(j, str(error_rate(x_iter,A,b)))
            x_iter = x_iter + alpha * gradient(b[j], A[j,:], x_iter)
            #print '%s %s' %(j, str(error_rate(x_iter,A,b)))
        curr_x = x_iter[:]
        obj_iter = evaluate_objective_value(b, A, curr_x)
        if objective_value_history[-1] > obj_iter:
            x_list.append(curr_x)
            objective_value_history.append(obj_iter)
        else:
            x_list.append(prev_x)
            objective_value_history.append(objective_value_history[-1])
            if objective_value_history[-1] != evaluate_objective_value(b, A, prev_x):
            #for i in range(len(prev_x)):
                print '%s %s' %(evaluate_objective_value(b, A, prev_x), objective_value_history[-1])
                sys.exit()
        #print x_list
        #print objective_value_history
        #if objective_value_history[-1] == objective_value_history[-2] and objective_value_history[-2] == objective_value_history[-3]:
            #print 'objective does not decrease'
            #sys.exit()
            #break
        if i in iter_list:
            tend = datetime.datetime.now()
            time_diff_list.append((tend-tstart).total_seconds() * 1000)
            print 'iteration %s time ' %(i),
            print tend
            training_error_list.append(error_count(x_list[-1], A, b, 'training'))
            testing_error_list.append(error_count(x_list[-1], A_test, b_test, 'testing'))
            #tstart = tend
    #print objective_value_history
    print objective_value_history[-1], len(objective_value_history)
    plot_classifier(A, x_list[-1], 'PLA-'+str(alpha), fig_line, alpha)
    plot_objective_evolution(fig_objective, objective_value_history, 'pocket algorithm', alpha)
    plot_error(color, fig_error, 'PLA-'+str(alpha), time_diff_list, training_error_list, testing_error_list)
    
    
#def plot_objective(objective_value_history, alpha):
#    iteration = []
#    for i in range(len(objective_value_history)):
#        iteration.append(i)
#    
#    pylab.plot(iteration, objective_value_history, label='stepsize '+str(alpha))
    
    
def evaluate_objective_value(b, A, x):
    objective_value = 0
    for j in range(A.shape[0]):
        #print objective_value
        if -1 * b[j] * np.dot(A[j,:], x) > 0:
            objective_value = objective_value + (-1 * b[j] * np.dot(A[j,:], x))
        elif -1 * b[j] * np.dot(A[j,:], x) < 0:
            objective_value = objective_value + 0
        else:
            print 'Got a false value when calculating objective value in pocket algorithm'
            sys.exit()
            
    return objective_value

def gradient(b_scalar, A_row, x):
    #print b_scalar
    #print A_row.transpose()
    #print x
    if -1 * b_scalar * np.dot(A_row, x) > 0:
        #print 'misclassified'
        #print b_scalar * A_row.transpose()
        #print b_scalar * A_row
        return b_scalar * A_row
    elif -1 * b_scalar * np.dot(A_row, x) < 0:
        return 0 * A_row
    else:
        print 'Got a false value when calculating gradient in pocket algorithm'
        sys.exit()
