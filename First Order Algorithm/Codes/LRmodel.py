from plot import *
import math

def logistic_regression(A, b, x_init, A_test, b_test, max_iter, digit, fig_line, fig_title):
    #print x_init
    print 'Solving logistic regression model with gradient descent method...'
    fig_objective = fig_title+' (logistic regression)'
    create_objective_plot(fig_objective)
    step_size = LRstep_size(A, b)
    converged = False
    iteration = 0
    x = x_init[:]
    x_list = []
    x_list.append(x)
    obj_list = []
    obj_list.append(LRobjective_value(A, b, x))
    while not converged:
        iteration += 1
        
        #print x
        #print LRgradient(A, b, x)
        #print rescaling(A, b, x)
        x = x - step_size * LRgradient(A, b, x)
        #x = x - step_size * np.dot(rescaling(A, b, x), LRgradient(A, b, x))
        x_list.append(x.tolist())
        obj_list.append(LRobjective_value(A, b, x))
        if iteration == max_iter:
            converged = True
        elif obj_list[-1] == obj_list[-2] and obj_list[-2] == obj_list[-3]:
            converged = True
    plot_objective_evolution(fig_objective, obj_list, 'logistic regression-'+str(round(step_size, 3)), step_size)
    plot_classifier(A, x, 'logistic regression- '+str(round(step_size, 3)), fig_line, step_size)
    objective_plot_define(fig_objective, max_iter, ylim = 90, xlim = 0)
    LRerror(x, A, b, 'training')
    LRerror(x, A_test, b_test, 'testing')
    #print obj_list
    print obj_list[-1], len(obj_list)
    print

def rescaling(A, b, x):
    rescaling_D = np.zeros((len(x), len(x)))
    for i in range(rescaling_D.shape[0]):
        for j in range(A.shape[0]):
            rescaling_D[i][i] = rescaling_D[i][i] + b[j] * A[j][i] * A[j][i] * math.exp(b[j] * A[j][i] * x[i]) / math.pow((1 + math.exp(b[j] * A[j][i] * x[i])), 2)
    #print rescaling_D
    #print np.linalg.inv(rescaling_D / A.shape[0])
    return np.linalg.inv(rescaling_D / A.shape[0])
    
def LRobjective_value(A, b, x):
    #print x
    objective_value = 0.0
    for i in range(A.shape[0]):
        objective_value += math.log1p(math.exp(-1 * b[i] * np.dot(A[i,:], x)))
        #objective_value += math.exp(-1 * b[i] * np.dot(A[i,:] * x))
    return objective_value / A.shape[0]

def LRgradient(A, b, x):
    gradient = np.zeros((len(x)))
    for i in range(A.shape[0]):
        gradient += (b[i] * A[i,:]) / (1 + math.exp(b[i] * np.dot(A[i,:], x)))
    return gradient / A.shape[0] * -1.0

def LRstep_size(A, b):
    stepsize = 0.0
    for i in range(A.shape[0]):
        stepsize += np.sum(A[i,:] ** 2)
    stepsize = stepsize / A.shape[0]
    #print 'step size = %s' %(stepsize)
    return 1.0/stepsize

def LRerror(x, A, b, error_type):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    error = 0
    for i in range(A.shape[0]):
        if b[i] == 1:
            if probability(A[i,:], x) > 0:
                true_positive += 1
            elif probability(A[i,:], x) == 0:
                print 'Got a false prediction'
                sys.exit()
            else:
                false_negative += 1
                error += 1
        elif b[i] == -1:
            if probability(A[i,:], x) > 0:
                false_positive += 1
                error += 1
            elif probability(A[i,:], x) == 0:
                print 'Got a false prediction'
                sys.exit()
            else:
                true_negative += 1
        else:
            print 'Got a false b value'
            sys.exit()
    print '%s error rate = %s%%, true positive = %s%%, true negative = %s%%, false positive = %s%%, false negative = %s%%' %(error_type, round(float(error)/A.shape[0]*100, 2), round(float(true_positive)/A.shape[0]*100,2), round(float(true_negative)/A.shape[0]*100,2), round(float(false_positive)/A.shape[0]*100,2), round(float(false_negative)/A.shape[0]*100,2))

def probability(a, x):
    return math.exp(np.dot(a, x)) / (1 + math.exp(np.dot(a, x)))
