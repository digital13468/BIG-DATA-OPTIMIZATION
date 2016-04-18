# -*- coding: utf-8 -*-
import numpy as np
import sys
from math import sqrt
import os
import csv
import pylab

def poweig(mat, x0, maxit = 100, ztol = 1.0e-6):
    result = x0
    xim1 = []
    xim1.append(x0.tolist())
    xim2 = []
    xim2.append(np.linalg.norm(result))
    #print 0, xim1[-1], xim2[-1]
    print 'iteration: ', 0, xim2[-1]
    print np.reshape(xim1[-1], (1, len(xim1[-1])))
    for i in xrange(maxit):
        result = mat * result
        tmp_eigval = np.linalg.norm(result)
        result = result/tmp_eigval
        xim1.append(result.tolist())
        xim2.append(tmp_eigval)
        if xim1[-1] != result.tolist():
            print 'Error'
            sys.exit()
        else:
            print 'iteration: ', i+1, xim2[-1]
            print np.reshape(xim1[-1], (1, len(xim1[-1])))
        #if sum([xim1[-1][k] - xim1[-2][k] for k in xrange[len(xim1[-1])]]) < ztol:
        if np.linalg.norm(np.asarray(xim1[-1]) - np.asarray(xim1[-2])) < ztol and sqrt((xim2[-2] - xim2[-1])**2) < ztol and i > 2:
            #print "%e" %np.linalg.norm(np.asarray(xim1[-1]) - np.asarray(xim1[-2]))
            #print  (np.asarray(xim1[-2]) - np.asarray(xim1[-1]))
            
            #print xim1[-1]
            #print xim1[-2]
            #print xim2[-2]
            #print xim2[-1]
            break
    prd = mat * result
    #if sqrt((prd[0]/result[0] - tmp_eigval)**2) >= ztol:
    #    print "Eigenvalue Different: ", tmp_eigval, (prd[0]/result[0])
    #    sys.exit()
    #else:
    #    if sqrt((prd[1]/result[1] - tmp_eigval)**2) >= ztol:
    #        print 'Different Factors', tmp_eigval, (prd[0]/result[0]), (prd[1]/result[1])
    #        sys.exit()
    #    else:
    print 'Computed Eigenvalue: ', tmp_eigval
    return result, tmp_eigval, xim1

def check(abs_eigvals, otp_eigval, rem_eigvals, ztol = 1.0e-5):
    
    #[eigs, vecs] = np.linalg.eig(mat)
    #abseigs = list(abs(eigs))
    ind = abs_eigvals.index(max(rem_eigvals))
    if sqrt((abs_eigvals[ind] - otp_eigval)**2) >= ztol:
        print 'Eigenvalues Not Match'
        print 'Largest Eigenvalue: ', abs_eigvals[ind]
        print 'Computed Eigenvalue: ', otp_eigval
        print 'Difference: ', sqrt((abs_eigvals[ind] - otp_eigval)**2)
        sys.exit()
    else:
        print 'Largest Eigenvalue: ', abs_eigvals[ind]
        print 'Difference: ', abs_eigvals[ind]-otp_eigval
    rem_eigvals.remove(max(rem_eigvals))

def get_data(filename):
    print 'Reading in Data'
    #f = open(os.path.join(filename),'r')
    #data = np.genfromtxt(filename)
    data = np.loadtxt(open(filename,'r'), delimiter = ',', skiprows = 1)
    #f = open(filename, 'r')
    #for row in csv.reader(f):
    #    print row
        
    #f.close()
    #print data
    return data.T

def plot_error(comp, banch, pc_ind):
    error = []
    iter_list = []
    for i in xrange(len(comp)):
        error.append(np.linalg.norm(comp[i] - banch))
       # print comp[i]
       # print banch
       # print comp[i]-banch


    
        iter_list.append(i)
    pylab.plot(iter_list, error, label = 'PC'+str(pc_ind+1))
    pylab.xlabel('Iteration')
    pylab.ylabel('Error from Standard Function (2-norm)')
    print 'error: '
    print np.reshape(error, (1, len(error)))
    pylab.legend()

def semantic(pc, word, num_ind = 20):
    
    #for i in xrange(len(pc)):
    #    sorted_pc = sorted(pc[i])
    #    print 'PC' + str(i+1) + ': '
    #    for j in xrange(-1, -1-num_ind, -1):
    #        print 'value, index and word: ', sorted_pc[j], pc[i].index(sorted_pc[j]), word[pc[i].index(sorted_pc[j])]
    #    for j in xrange(-1, -1-num_ind, -1):
    #        
    #        frequency = abs(np.asarray(sorted_pc[j])/np.asarray(sorted_pc[-num_ind]))
    #        #print np.asarray(sorted_pc[j]),np.asarray(sorted_pc[-num_ind]),int(round(frequency))
    #        for k in xrange(int(round(frequency))):
    #            print word[pc[i].index(sorted_pc[j])],
    
    for i in xrange(len(pc)):
        abs_pc = np.absolute(pc[i]).tolist()
        sorted_pc = sorted(abs_pc)
        print 'PC' + str(i+1) + ': '
        for j in xrange(-1, -1-num_ind, -1):
            print 'value, index and word: ', pc[i][abs_pc.index(sorted_pc[j])], abs_pc.index(sorted_pc[j]), word[abs_pc.index(sorted_pc[j])]
        #factor = 1.0/sorted_pc[0]
        for j in xrange(-1, -1-num_ind, -1):
            
            frequency = np.asarray(sorted_pc[j])/np.asarray(sorted_pc[-num_ind])
            #print np.asarray(sorted_pc[j]),np.asarray(sorted_pc[-num_ind]),int(round(frequency))
            for k in xrange(int(round(frequency))):
                print word[abs_pc.index(sorted_pc[j])],
        print
        
if __name__== "__main__":
    matrix = get_data('20news_w100.csv')
    #matrix = [[1, 0.296], [0.296, 1]]
    #x = np.ones((2,1))
    #matrix =[[2,-1,1,-2],[-1,1,0,1],[1,0,1,-1],[-2,1,-1,2]]
    #matrix = [[-4, 10], [7, 5]]
    #matrix = [[4, 1, 0], [0, 2, 1], [0, 0, -1]]
    #print len(matrix)
    
    #matrix = [[2,0,1,1,0],[0,1,1,0,0],[1,1,2,1,0],[1,0,1,3,1],[0,0,0,1,2]]
    x = np.ones((len(matrix),1))
    nbit = 300
    eigvecs = []
    eigvals = []
    #x = np.random.random((len(matrix),1))
    A = np.dot(matrix, matrix.T)
    #A = np.matrix(matrix)
    #print A
    #A[0][0]=1
    #print A
    #print matrix
    [eigs, vecs] = np.linalg.eig(A)
    abseigs = list(abs(eigs))
    eigs = abseigs[:]
    #for i in xrange(len(A)):
    for i in xrange(4):
        print 'Finding %dth Latgest Eigens for' % (i+1)
        print A
        x, xlambda, x_list = poweig(A, np.matrix(x), nbit)
        #if i == 0:
        #    plot_error(x_list, vecs[:, eigs.index(max(eigs))])
        if i < 2:
            plot_error(x_list, vecs[:, i], i)                                    
        eigvecs.append(x.tolist())
        eigvals.append(xlambda)
        check(abseigs, xlambda, eigs)
        
        print "poweig returns"
        print x.T
        print xlambda
        if i > 0:
            print 'verification: ', np.transpose(np.matrix(eigvecs[-1])) * np.matrix(eigvecs[-2])
        A = A - xlambda * np.dot(x, np.transpose(x))
        print
    print np.reshape(eigvecs, (len(eigvecs),len(eigvecs[0])))
    
    print eigvals
    print vecs.T
    print abseigs
    
    semantic(eigvecs, csv.reader(open("20news_w100.csv")).next())
    pylab.show()
