#!/usr/bin/python

import sys
#import os
import numpy
#import scipy
import scipy.spatial as scip
#import re
import time
import random

NUMBER_OF_TRAINING_INSTANCES = 15

#Compute DTW Distance
def dtwdis(temp,inp):
    # Use eclidean / mahalanobi
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
        trl=range(1,NUMBER_OF_TRAINING_INSTANCES)
        random.shuffle(trl)
        animal_array = ['dog', 'cat']
        for i in range(0,2):
            cos=[]
            for m in trl:
                temdat=numpy.loadtxt('animal_features/'+animal_array[i]+'_'+str(m)+'.mfcc')
                fcost=dtwdis(temdat,inpdat)
                print '{0}_{1}  distance from input : --->  {2}'.format(animal_array[i],m,fcost)
                cos[len(cos):]=[fcost]
            print '\n'
            cost[len(cost):]=[min(cos)]
        
        #print '\n'
        animal_index = cost.index(min(cost))
        animal = "dog" if animal_index == 0 else "cat";
        print 'Animal recognised as {0}'.format(animal)
        et=time.clock()
        print '{0} seconds'.format(et-st)


if __name__ == '__main__':
    main(sys.argv)
