#!/usr/bin/python

import sys
#import os
import numpy
#import scipy
import scipy.spatial as scip
#import re
import time
import random

def dtwdis(temp,inp):
    dmat=scip.distance.cdist(temp,inp,'euclidean')
    m,n=numpy.shape(dmat)
    
    dcost=numpy.ones((m+2,n+1))
    dcost=dcost+numpy.inf

    dcost[2,1]=dmat[0,0]
    k=3
    for j in range(2,n+1):
       for i in range(2,min(2+k,m+2)):
           dcost[i,j]=min(dcost[i,j-1],dcost[i-1,j-1],dcost[i-2,j-1])+dmat[i-2,j-1]
       k=k+2

    return(dcost[m+1,n])

def main(argv):
    
    if argv[1]=='-d':
        st=time.clock()
        inpdat=numpy.loadtxt(argv[2])
        cost=[]
        trl=range(0,5)
        random.shuffle(trl)
        trlist=trl[0:int(argv[3])]
        for i in range(0,10):
            cos=[]
            for m in trlist:
                temdat=numpy.loadtxt('all_recs/'+str(i)+'_'+str(m)+'.mfcc')
                fcost=dtwdis(temdat,inpdat)
                print '{0}_{1}---{2}'.format(i,m,fcost)
                cos[len(cos):]=[fcost]
            print '\n'
            cost[len(cost):]=[min(cos)]
        
        #print '\n'
        print 'Digit recognised as {0}'.format(cost.index(min(cost)))
        et=time.clock()
        print et-st

    elif argv[1]=='-r':
        st=time.clock()
        tlist=[0,1,2,3,4]
        #tlist=[5,6,7,8,9]

        trlist=range(5,5+int(argv[2]))
        #trlist=range(0,0+int(argv[2]))

        #trl=range(5,10)
        #random.shuffle(trl)
        #trlist=trl[0:int(argv[2])]
        ac=0.0
        for i in range(0,10):
            for l in tlist:
                inpdat=numpy.loadtxt('all_recs/'+str(i)+'_'+str(l)+'.mfcc')
                cost=[]
                for j in range(0,10):
                    cos=[]
                    for m in trlist:
                        temdat=numpy.loadtxt('all_recs/'+str(j)+'_'+str(m)+'.mfcc')
                        fcost=dtwdis(temdat,inpdat)
                        cos[len(cos):]=[fcost]
                    
                    cost[len(cost):]=[min(cos)]
                
                print '{0}_{1} recognised as {2}'.format(i,l,cost.index(min(cost)))
                if i==cost.index(min(cost)):
                    ac=ac+1.0
            print '\n'
        print 'Recognition Accuracy {0}'.format(ac/50)
        et=time.clock()
        print et-st
if __name__ == '__main__':
    main(sys.argv)
