import numpy as np
import random
from plot import *
import datetime

def PBCGD(block_size, A, b, A_test, b_test, max_iter, x_init, fig_objective, fig_line):
    x = x_init[:]
    print '   ======================   %s - %s   ======================' % ('PBCGD', str(block_size))
    x_list = []
    obj_list = []
    converged = False
    iter = 0
    x_list.append(x)
    obj_list.append(np.sum((np.dot(A, x) - b) ** 2) / 2)
    #block_start = 0
    #print x
    while not converged:
        block_index = np.zeros(block_size, dtype=np.int)
        iter += 1
        block_start = random.randint(0, len(x)-block_size)
        for i in range(block_size):
            block_index[i] = block_start + i
        #if block_index[-1] == len(x) - 1:
        #    print block_index
        for k in range(block_size):
            subproblem_A = A[:,block_index[k]]
            otherproblem_A = np.zeros(A.shape[0])
            Hessian_L = np.dot(A[:,block_index[k]], A[:,block_index[k]])
            for i in range(A.shape[1]):
                #for j in range(block_index.shape[0]):
                if i != block_index[k]:
                    #print otherproblem_A
                    #print x[i]
                    otherproblem_A = otherproblem_A + A[:,i] * x[i]
            #print x_init
            x[block_index[k]] = np.dot(subproblem_A, (b - otherproblem_A)) / Hessian_L
        x_list.append(x)
        obj_list.append(np.sum((np.dot(A, x) - b) ** 2) / 2)
        if iter == max_iter:
            converged = True
        elif obj_list[-1] == obj_list[-2] and obj_list[-2] == obj_list[-3]:
            converged = True
        
        #print obj_list[-1]
    error_count(x_list[-1], A, b, 'training')
    error_count(x_list[-1], A_test, b_test, 'testing')
    #print obj_list
    print obj_list[-1], len(obj_list)
    print
    plot_objective_evolution(fig_objective, obj_list, 'PBCGD')
    plot_classifier(A, x, 'PBCGD-'+str(block_size), fig_line)
            
def CBCGD(block_size, color, fig_error, A, b, A_test, b_test, max_iter, x_init, fig_objective, fig_line):
    print '   ======================   %s - %s   ======================' % ('CBCGD', str(block_size))
    x_list = []
    x = x_init[:]
    converged = False
    iter = 0
    x_list.append(x)
    obj_list = [np.sum((np.dot(A, x) - b) ** 2) / 2]
    block_start = 0
    #print x
    iter_list = create_iter_list(max_iter)
    training_error_list = []
    testing_error_list = []
    time_diff_list = []
    tstart = datetime.datetime.now()
    while not converged:
       # print obj_list
        #print x
        block_index = np.zeros(block_size, dtype=np.int)
        iter += 1
        for i in range(block_size):
            block_index[i] = (block_start + i) % len(x)
        for k in range(block_size):
            subproblem_A = A[:,block_index[k]]
            otherproblem_A = np.zeros(A.shape[0])
            Hessian_L = np.dot(A[:,block_index[k]], A[:,block_index[k]])
            for i in range(A.shape[1]):
                #for j in range(block_index.shape[0]):
                if i != block_index[k]:  
                    otherproblem_A = otherproblem_A + A[:,i] * x[i]
            #print '%s %s %s' %(k, block_index[k], x[block_index[k]])
            x[block_index[k]] = np.dot(subproblem_A, (b - otherproblem_A)) / Hessian_L
        x_list.append(x)
        obj_list.append(np.sum((np.dot(A, x) - b) ** 2) / 2)
        #print obj_list
        #print block_index
        if iter == max_iter:
            converged = True
        #elif obj_list[-1] == obj_list[-2] and obj_list[-2] == obj_list[-3]:
        #    converged = True
        block_start = (block_size + block_start) % len(x)

        if iter in iter_list:
            tend = datetime.datetime.now()
            print 'iteration %s time ' %(iter),
            print tend
            training_error_list.append(error_count(x_list[-1], A, b, 'training'))
            time_diff_list.append((tend-tstart).total_seconds()*1000)
            testing_error_list.append(error_count(x_list[-1], A_test, b_test, 'testing'))
            tstart = tend
    #error_count(x_list[-1], A, b, 'training')
    #error_count(x_list[-1], A_test, b_test, 'testing')
    #print obj_list
    print obj_list[-1], len(obj_list)
    print
    plot_objective_evolution(fig_objective, obj_list, 'CBCGD')
    plot_classifier(A, x, 'CBCGD-'+str(block_size), fig_line)
    plot_error(color, fig_error, 'CBCGD', time_diff_list, training_error_list, testing_error_list)
    
def RBCGD(block_size, color, fig_error, A, b, A_test, b_test, max_iter, x_init, fig_objective, fig_line):
    print '   ======================   %s - %s   ======================' % ('RBCGD', str(block_size))
    x = x_init[:]
    x_list = []
    obj_list = []
    converged = False
    iter = 0
    x_list.append(x)
    obj_list.append(np.sum((np.dot(A, x) - b) ** 2) / 2)
    #block_start = 0
    time_diff_list = []
    iter_list = create_iter_list(max_iter)
    training_error_list = []
    tstart = datetime.datetime.now()
    testing_error_list = []
    time_diff_list.append(0)
    print 'iteration 0 at ',
    print tstart
    training_error_list.append(error_count(x_list[-1], A, b, 'training'))
    testing_error_list.append(error_count(x_list[-1], A_test, b_test, 'testing'))
    while not converged:
        block_index = np.zeros(block_size, dtype=np.int)
        iter += 1
        #blcok_start = random.randint(0, len(x)-block_size)
        for i in range(block_size):
            block_index[i] = random.randint(0, len(x)-1)
        #print block_index
        for k in range(block_size):
            subproblem_A = A[:,block_index[k]]
            otherproblem_A = np.zeros(A.shape[0])
            Hessian_L = np.dot(A[:,block_index[k]], A[:,block_index[k]])
            for i in range(A.shape[1]):
                #for j in range(block_index.shape[0]):
                if i != block_index[k]:
                    #print otherproblem_A
                    #print x[i]
                    otherproblem_A = otherproblem_A + A[:,i] * x[i]
            #print '%s %s %s' %(k, block_index[k], x[block_index[k]])
            x[block_index[k]] = np.dot(subproblem_A, (b - otherproblem_A)) / Hessian_L
        x_list.append(x)
        obj_list.append(np.sum((np.dot(A, x) - b) ** 2) / 2)
        if iter == max_iter:
            converged = True
        #elif obj_list[-1] == obj_list[-2] and obj_list[-2] == obj_list[-3]:
        #    converged = True
        #print block_index
        #print obj_list[-1]
        if iter in iter_list:
            tend = datetime.datetime.now()
            print 'iteration %s time ' %(iter),
            print tend
            training_error_list.append(error_count(x_list[-1], A, b, 'training'))
            time_diff_list.append((tend-tstart).total_seconds()*1000)
            testing_error_list.append(error_count(x_list[-1], A_test, b_test, 'testing'))
            #tstart = tend
   # error_count(x_list[-1], A, b, 'training')
    #error_count(x_list[-1], A_test, b_test, 'testing')
    #print obj_list
    print obj_list[-1], len(obj_list)
    print
    plot_objective_evolution(fig_objective, obj_list, 'RBCGD')
    plot_classifier(A, x, 'RBCGD-'+str(block_size), fig_line)
    plot_error(color, fig_error, 'RBCGD', time_diff_list, training_error_list, testing_error_list)
