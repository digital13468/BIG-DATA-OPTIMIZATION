import pylab
import numpy as np
import math
import sys

def create_iter_list(max_iter):
    L = []
    for i in range(0, max_iter+1, 1):
        
        L.append(i)
    return L

def error_count(x, A, b, error_type):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    error = 0
    for i in range(A.shape[0]):
        if b[i] == 1:
            if np.dot(A[i,:], x) > 0:
                true_positive += 1
            elif np.dot(A[i,:], x) == 0:
                print 'Got a false prediction'
                sys.exit()
            else:
                false_negative += 1
                error += 1
        elif b[i] == -1:
            if np.dot(A[i,:], x) > 0:
                false_positive += 1
                error += 1
            elif np.dot(A[i,:], x) == 0:
                print 'Got a false prediction'
                sys.exit()
            else:
                true_negative += 1
        else:
            print 'Got a false b value'
            sys.exit()
    print '%s error rate = %s%%, true positive = %s%%, true negative = %s%%, false positive = %s%%, false negative = %s%%' %(error_type, round(float(error)/A.shape[0]*100, 2), round(float(true_positive)/A.shape[0]*100,2), round(float(true_negative)/A.shape[0]*100,2), round(float(false_positive)/A.shape[0]*100,2), round(float(false_negative)/A.shape[0]*100,2))
    return float(error)/A.shape[0]*100
    
def create_objective_plot(fig_name):
    pylab.figure(fig_name)
    #pylab.title(fig_name)
    pylab.xlabel('Number of Iteration')
    pylab.ylabel('Objective Value')

def objective_plot_define(fig_name, max_iter, ylim = 0, xlim = 0):
    pylab.figure(fig_name)
    pylab.legend()
    if (ylim != 0):
        pylab.ylim(0, ylim)
    if (xlim != 0):
        pylab.xlim((xlim, max_iter))
    
    pylab.savefig(fig_name+'.eps')
    

def plot_objective_evolution(fig_name, obj_list, rule, alpha = 0, s = 0, beta = 0, sigma = 0):
    #plot_classifier(A, b, x_list[-1], 'GD')
    pylab.figure(fig_name)
    
    iter_list = []
    for i in range(len(obj_list)):
        iter_list.append(i)
    #pylab.subplot(7, 1, fig[0])

    #fig[0] = fig[0] + 1
    #if obj_list[-1] == np.inf:
    #    pylab.annotate('('+str(iter_list[-1])+', diverged)', xy=(iter_list[-4], obj_list[-4]), xytext=(iter_list[-4]*0.8, obj_list[-4]*0.8), arrowprops=dict(facecolor='black', shrink=0.03))
    #else:
    #    pylab.annotate('('+str(iter_list[-1])+', '+str(round(obj_list[-1], 3))+')', xy=(iter_list[-1], obj_list[-1]), xytext=(iter_list[-1]*0.8, obj_list[1]), arrowprops=dict(facecolor='black', shrink=0.03))
    #pylab.plot(iter_list, obj_list, lw=3)
    #pylab.xlabel('Number of Iteration')
    #pylab.ylabel('Objective Value')
    if (rule == 'constant'):
        #pylab.title(rule + '-' +str(round(alpha, 6)))
        pylab.plot(iter_list, obj_list, label=rule + '-' +str(round(alpha, 6)))
    elif (rule == 'armijo'):
        #pylab.title(rule + '(s/beta/sigma) - ' + str(round(s, 3)) + '/' + str(round(beta, 3)) + '/' +str(round(sigma, 3)))
        pylab.plot(iter_list, obj_list, label=rule + '(s/beta/sigma) - ' + str(round(s, 3)) + '/' + str(round(beta, 3)) + '/' +str(round(sigma, 3)))
    else:
        #pylab.title(rule)
        pylab.plot(iter_list, obj_list, label=rule)

def plot_error(color, fig_name, legend_name, iter_list, training_list, testing_list):
    pylab.figure(fig_name)
    pylab.plot(iter_list, training_list, color[0]+':', label=legend_name+'-training')
    pylab.plot(iter_list, testing_list, color[0], label=legend_name+'-testing')
    color.remove(color[0])
    
def error_plot_define(fig_name, max_iter = 0, ylim = 0, xlim = 0):
    pylab.figure(fig_name)
    pylab.legend()
    #if (ylim != 0):
    #    pylab.ylim(0, ylim)
    #if (xlim != 0):
    #    pylab.xlim((xlim, max_iter))
    #pylab.tight_layout()
    pylab.savefig(fig_name+'.eps')
    
    
def create_error_plot(fig_name):
    pylab.figure(fig_name)
    #pylab.title(fig_name)
    pylab.xlabel('Number of passes')
    pylab.ylabel('Error rate (%)')
    
def create_line_plot(fig_name, A, b):
    pylab.figure(fig_name)
    #pylab.title(fig_name)
    for i in range(b.shape[0]):
        
        
        if b[i] == 1:
            pylab.plot(A[i,0], A[i,1], 'b*')
        
        else:
            pylab.plot(A[i,0], A[i,1], 'r.')

def line_plot_define(fig_name):
    pylab.figure(fig_name)
    pylab.xlabel('Intensity')
    pylab.ylabel('Symmetry')
    pylab.legend()
    #pylab.tight_layout()
    pylab.savefig(fig_name+'.eps')
    

def plot_classifier(A, x, rule, fig_name, alpha = 0, s = 0, beta = 0, sigma = 0, choose = 0):
    pylab.figure(fig_name)
    predict = []
    
    for i in range(A.shape[0]):
        #print '%s %s %s %s' %(A[i,0], x[0], x[2], x[1])
        predict.append(-1*((A[i,0]*x[0]+x[2]*A[i,2])/x[1]))
        
    #print predict
    if (rule == 'constant'):
        #pylab.title(rule + '-' +str(round(alpha, 6)))
        pylab.plot(A[:,0], predict, '--', label=rule + '-' +str(round(alpha, 6)))
    elif (rule == 'armijo'):
        #pylab.title(rule + '(s/beta/sigma) - ' + str(round(s, 3)) + '/' + str(round(beta, 3)) + '/' +str(round(sigma, 3)))
        pylab.plot(A[:,0], predict, '-.', label=rule + '(s/beta/sigma) - ' + str(round(s, 3)) + '/' + str(round(beta, 3)) + '/' +str(round(sigma, 3)))
    elif choose == 2:
        pylab.plot(A[:,0], predict, ':' , label=rule)
    else:
        #pylab.title(rule)    
        pylab.plot(A[:,0], predict, label=rule)
    
