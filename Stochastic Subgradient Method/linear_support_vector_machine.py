import numpy as np
from plot import *
import sys
import os
import datetime

def linear_SVM(color, A, b, x_init, max_iter, digit, fig_error, fig_line, fig_title, A_test, b_test):
    print 'Solving linear SVM with stochastic subgradient algorithm...'
    model_constant = np.random.random(1)
    gamma = 1.0 / (2 * model_constant * A.shape[0])
    fig_objective = fig_title + ' (linear SVM)'
    create_objective_plot(fig_objective)
    stochastic_subgradient(color, b, A, x_init, max_iter, model_constant, gamma, fig_error, fig_line, fig_objective, A_test, b_test, 1)
    stochastic_subgradient(color, b, A, x_init, max_iter, model_constant, gamma, fig_error, fig_line, fig_objective, A_test, b_test, 1.0/2.0)
    
    objective_plot_define(fig_objective, max_iter)

def testing_stage(final_x, A, b, A_test, b_test):
    training_error = error_count(final_x, A, b, 'training')
    
    testing_error = error_count(final_x, A_test, b_test, 'testing')
    return training_error, testing_error

def stochastic_subgradient(color, b, A, x_init, max_iter, model_constant, gamma, fig_error, fig_line, fig_objective, A_test, b_test, step_rule):
    print('   ================   stochastic with alpha = %s/r^%s   ================   ') %(str(round(model_constant, 3)), str(step_rule))
    x_list = []
    x_list.append(x_init[:])
    objective_value_history = []
    objective_value_history.append(linear_SVM_objective(b, A, x_init[:], gamma))
    x_iter = x_init[:]
    iter_list = create_iter_list(max_iter)
    training_error_list = []
    testing_error_list = []
    time_diff_list = []
    tstart = datetime.datetime.now()
    time_diff_list.append(0)
    print 'iteration 0 at ',
    print tstart
    training_error, testing_error = testing_stage(x_list[-1], A, b, A_test, b_test)
    training_error_list.append(training_error)
    testing_error_list.append(testing_error)
    for i in range(1, max_iter+1):
        #prev_x = x_list[-1][:]
        data_point = np.random.randint(A.shape[0])
        x_iter = x_iter - (model_constant / i ** step_rule) * subgradient(b[data_point], A.shape[0], gamma, x_iter[:], A[data_point,:])
        if objective_value_history[-1] >= linear_SVM_objective(b, A, x_iter.tolist(), gamma):
            x_list.append(x_iter.tolist())
            objective_value_history.append(linear_SVM_objective(b, A, x_iter.tolist(), gamma))
            #if objective_value_history[-1] != linear_SVM_objective(b, A, x_list[-1], gamma):
            if np.isclose(objective_value_history[-1], linear_SVM_objective(b, A, x_list[-1], gamma)) == False:
                print 'Access error'
                print x_list[-1]
                print x_iter
                print x_iter.tolist()
                print objective_value_history[-1]
                print linear_SVM_objective(b, A, x_list[-1], gamma)
                print linear_SVM_objective(b, A, x_iter, gamma)
                sys.exit()
        else:
        #    print 'Objective increased at iteration %s from %s to %s with %s' %(i, objective_value_history[-1], linear_SVM_objective(b, A, x_iter, gamma), linear_SVM_objective(b, A, x_iter, gamma) - objective_value_history[-1])
            #os.system('pause')
            #x_list.append(x_list[-1][:])
            #objective_value_history.append(linear_SVM_objective(b, A, x_list[-1][:], gamma).tolist())
            x_list.append(x_iter.tolist())
            objective_value_history.append(linear_SVM_objective(b, A, x_iter.tolist(), gamma))
            #print 'Objective increased at iteration %s from %s to %s with %s' %(i, objective_value_history[-2], objective_value_history[-1], objective_value_history[-1] - objective_value_history[-2])
            #if objective_value_history[-1] != linear_SVM_objective(b, A, x_list[-1], gamma):
            if np.isclose(objective_value_history[-1], linear_SVM_objective(b, A, x_list[-1], gamma)) == False:
                print 'Access error'
                print x_list[-1]
                print x_iter
                print x_iter.tolist()
                print objective_value_history[-1]
                print linear_SVM_objective(b, A, x_list[-1], gamma)
                print linear_SVM_objective(b, A, x_iter, gamma)
                sys.exit()
        #print 'iter %s | objective: %s | x: %s | alpha: %s ' % (i, objective_value_history[-1], x_list[-1], (model_constant / i ** step_rule)),
        #if objective_value_history[-1] > objective_value_history[-2]:
        #    print '| increased by: '+ str(objective_value_history[-1] - objective_value_history[-2])
            #os.system('pause')
        #elif objective_value_history[-1] == objective_value_history[-2] and objective_value_history[-3] == objective_value_history[-2]:
        #    print 'objective values stay the same'
            #os.system('pause')
        #else:
            #print
        if i in iter_list:
            tend = datetime.datetime.now()
            time_diff_list.append((tend-tstart).total_seconds()*1000)
            print 'iteration %s time ' %(i),
            print tend
            training_error, testing_error = testing_stage(x_list[-1], A, b, A_test, b_test)
            training_error_list.append(training_error)
            testing_error_list.append(testing_error)
            #tstart = tend
    #print x_list
    print objective_value_history[-1], len(objective_value_history)
    plot_classifier(A, x_list[-1], 'linear SVM (alpha= '+str(round(model_constant, 3))+'/r^'+str(step_rule)+')', fig_line)
    plot_objective_evolution(fig_objective, objective_value_history, 'Stochastic subgradient method (alpha= '+str(round(model_constant, 3))+'/r^'+str(step_rule)+')')
    plot_error(color, fig_error, 'Linear SVM (alpha= '+str(round(model_constant, 3))+'/r^'+str(step_rule)+')', time_diff_list, training_error_list, testing_error_list)

def subgradient(b_scalar, dataset_size, gamma, x, A_row):
    if b_scalar * np.dot(A_row, x) <= 1:
        return (2 * gamma * x) - (b_scalar * A_row / dataset_size)
    elif b_scalar * np.dot(A_row, x) > 1:
        return 2 * gamma * x
    else:
        print 'Got a false value when calculating subgradient'
        sys.exit()
        
def linear_SVM_objective(b, A, x, gamma):
    objective_value = 0
    for i in range(A.shape[0]):
        if 1 - b[i] * np.dot(A[i,:], x) >= 0:
            objective_value += 1 - b[i] * np.dot(A[i,:], x)
        elif 1 - b[i] * np.dot(A[i,:], x) < 0:
            continue
        else:
            print 'Got a false value when calculating linear SVM objective'
            sys.exit()
    return (objective_value / A.shape[0]) + (gamma * np.dot(x, x))
