import numpy as np
from plot import *
import sys
import os
import datetime
import random

def margin_SVM(color, A, b, max_iter, i, fig_error, fig_line, fig_title, A_test, b_test):
    print 'Solving margin SVM with dual problem...'
    fig_objective = fig_title + ' (dual SVM)'
    #c = random.random()
    create_objective_plot(fig_objective)
    initial_lambda = np.random.rand(A.shape[0]).tolist()
    #initial_lambda = np.ones(A.shape[0]).tolist()
    #print initial_lambda
    dual_problem(color, b, A, initial_lambda, max_iter, fig_error, fig_line, fig_objective, A_test, b_test, np.amax(initial_lambda))
    dual_problem(color, b, A, initial_lambda, max_iter, fig_error, fig_line, fig_objective, A_test, b_test, np.amax(initial_lambda) * math.pow(3, random.random()*10), choose = 2)
    objective_plot_define(fig_objective, max_iter)

def dual_problem(color, b, A, initial_lambda, max_iter, fig_error, fig_line, fig_objective, A_test, b_test, c, choose = 0):
    if c < np.amax(initial_lambda):
        print 'false c value'
        sys.exit()
    print('   ================   dual problem with c = %s   ================   ') %(str(round(c, 3)))
    lambda_list = []
    lambda_list.append(initial_lambda[:])
    dual_obj_list = []
    dual_obj_list.append(dual_objective_cal(lambda_list[-1], b, A))
    lamb_iter = lambda_list[-1][:]
    iter_list = create_iter_list(max_iter)
    training_error_list = []
    testing_error_list = []
    time_diff_list = []
    tstart = datetime.datetime.now()
    time_diff_list.append(0)
    print 'iteration 0 at ',
    print tstart
    x_list = []
    x_list.append(get_primal_value(lambda_list[-1], A, b))
    primal_obj_list = []
    primal_obj_list.append(get_primal_obj(lambda_list[-1], A, b, x_list[-1], c))
    training_error_list.append(error_count(x_list[-1], A, b, 'training'))
    testing_error_list.append(error_count(x_list[-1], A, b, 'testing'))
    for h in range(1, max_iter+1):
        for i in range(A.shape[0]):
            update = lamb_iter[i] + ((1.0 / (b[i] * b[i] * np.dot(A[i,:], A[i,:]))) * dual_gradient(lamb_iter, A, b, i))
            lamb_iter[i] = projection(update, c)
        lambda_list.append(lamb_iter)
        dual_obj_list.append(dual_objective_cal(lambda_list[-1], b, A))
        x_list.append(get_primal_value(lambda_list[-1], A, b))
        primal_obj_list.append(get_primal_obj(lambda_list[-1], A, b, x_list[-1], c))
        tend = datetime.datetime.now()
        time_diff_list.append((tend-tstart).total_seconds()*1000)
        print 'iteration %s time ' %(h),
        print tend
        training_error_list.append(error_count(x_list[-1], A, b, 'training'))
        testing_error_list.append(error_count(x_list[-1], A_test, b_test, 'testing'))
        if h == 1 or h == 5 or h == 10 or h ==50:
            if choose == 2:
                plot_classifier(A, x_list[-1], 'Dual SVM (c= '+str(round(c, 3))+ ')-' + str(h) + 'th pass', fig_line, choose = 2)
            else:
                plot_classifier(A, x_list[-1], 'Dual SVM (c= '+str(round(c, 3))+ ')-' + str(h) + 'th pass', fig_line)
    #check_constraints(x_list[-1], lambda_list[-1], A, b, c)    
    print dual_obj_list[-1], primal_obj_list[-1]
    #plot_classifier(A, x_list[-1], 'Dual SVM (c= '+str(round(c, 3))+ ')', fig_line)
    plot_objective_evolution(fig_objective, dual_obj_list, 'Dual method (c= '+str(round(c, 3))+')')
    plot_error(color, fig_error, 'Dual SVM (c = '+str(round(c, 3))+')', iter_list, training_error_list, testing_error_list)

def check_constraints(x, lamb, A, b, c):
    for i in range(A.shape[0]):
        if lamb[i] < 0 or lamb[i] > c:
            print 'lambda error'
            sys.exit()
        elif lamb[i] == 0:
            if b[i] * np.dot(A[i,:], x) < 1:
                print 'casi error'
                sys.exit()
        else:
            if b[i] * np.dot(A[i,:], x) > 1:
                print 'casi 2 error'
                sys.exit()
        
def projection(value, c):
    #projeced = 0
    if value > c:
        return c
    elif value < 0:
        return 0
    else:
        return value

def dual_gradient(lamb, A, b, i):
    gradient = 0
    summation = 0
    for j in range(A.shape[0]):
        summation += lamb[j] * b[j] * A[j,:]
    gradient = 1 - b[i] * np.dot(A[i,:], summation)
    return gradient

def get_primal_obj(lamb, A, b, x, c):
    casi = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        if lamb[i] >= 0:
            casi[i] = 1 - (b[i] * np.dot(A[i,:], x))
            #if casi[i] < 0:
            #    print 'casi error'
            #    print i
            #    sys.exit()
        #elif lamb[i] == 0:
        #    if b[i] * np.dot(A[i,:], x) < 1:
        #        print 'constraint violated'
        #        print lamb
        #        print x
        #        print np.dot(A[i,:], x)
        #        print b[i]
        #        print i
        #        sys.exit()
        else:
            print 'lambda negative'
            sys.exit()
    primal_objective = 0.5 * np.dot(x, x)
    for i in range(A.shape[0]):
        primal_objective += c * casi[i]
    return primal_objective

def get_primal_value(lamb, A, b):
    x = np.zeros(A.shape[1])
    for i in range(A.shape[0]):
        x += lamb[i] * b[i] * A[i,:]
    return x

def dual_objective_cal(lamb, b, A):
    objective_value = 0
    
    for i in range(A.shape[0]):
        objective_value += lamb[i]
    for i in range(A.shape[0]):    
        for j in range(A.shape[0]):
            objective_value += -1 * 0.5 * lamb[i] * lamb[j] * b[i] * b[j] * np.dot(A[i,:], A[j,:])
            
    return objective_value                
