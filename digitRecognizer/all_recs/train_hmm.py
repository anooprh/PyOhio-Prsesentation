import sys
import numpy as np
import scipy.spatial.distance

def do_DTW(HMM, trans_mat, data):
    means = HMM[::2,:]
    vars = HMM[1::2,:]

    #vars = vars+ 0.001

    DTW_dist = np.zeros((5,data.shape[0]))
    
    for i in xrange(5):
        inv_cov = np.linalg.inv(np.diagflat(vars[i][:]))
        tmp_dist = scipy.spatial.distance.cdist(np.matrix(means[i][:]),data,
                                      #          'euclidean')
                                                'mahalanobis',VI=inv_cov)
        DTW_dist[i][:] = 0.5*tmp_dist + 0.5*np.log(np.prod(vars[i][:])) #+ 19.5*np.log(2*np.pi)

    np.savetxt('dist_file',DTW_dist)
                                              

    # Do actual DTW: Anurag's code
    m,n = np.shape(DTW_dist)
    dcost = np.ones((m+2,n+1)) 
    dcost = dcost + np.inf

    DTW_bptr = np.zeros((m+2,n+1))
    DTW_bptr = DTW_bptr + np.inf
    

    dcost[2,1] = DTW_dist[0,0]

    k=3
    for j in range(2,n+1):
       for i in range(2,min(2+k,m+2)):
           costs = np.array([dcost[i,j-1]+trans_mat[i][0],
                            dcost[i-1,j-1]+trans_mat[i-1][1],
                            dcost[i-2,j-1]+trans_mat[i-2][2]])
           dcost[i,j] = np.min(costs) +DTW_dist[i-2,j-1]
           tmp_ptr = np.argmin(costs)

           if tmp_ptr == 0:
               DTW_bptr[i,j] = i
           elif tmp_ptr == 1:
               DTW_bptr[i,j] = i-1
           else:
               DTW_bptr[i,j] = i-2
       k=k+2

    np.savetxt('bptr_file',DTW_bptr)

    seg = np.zeros((4,1)) # 4 cuts

    prev=6.0
    current=6.0
    j = n
    btrace = np.zeros((n+1,))
    trans_count = np.zeros((5,5))

    btrace[0] = 2
    btrace[1] = 2
    while j >= 2:
        btrace[j] = prev
        current = DTW_bptr[prev][j]
        j = j - 1
        trans_count[current-2,prev-2] = trans_count[current-2,prev-2] + 1
        prev = current
        
    btrace = btrace -2

    binct = np.bincount(btrace.astype(np.int64,casting='unsafe'))


    prev=0
    for j in xrange(4): #Last cut does not matter
            seg[j] = binct[j] + prev
            prev = seg[j]

    tr_count = np.concatenate((np.matrix(trans_count[0,:3]),
                               np.matrix(trans_count[1,1:4]),
                               np.matrix(trans_count[2,2:5]),
                               np.matrix(np.append(trans_count[3,3:],0)),
                               np.matrix(np.append(np.append(trans_count[4,4],0),0))),
                              axis=0)

    tr_count = tr_count + 0.0001 #Avoiding infinity errors
    tr_count = tr_count/np.sum(tr_count)
    tr_count = np.log(tr_count)


    best_cost = dcost[dcost.shape[0]-1, 
                      dcost.shape[1]-1]

    return seg, tr_count, best_cost



if len(sys.argv) <= 1:
    print "Usage:\npython train_hmm.py digit"
    print "Run in directory with MFCCs\n"
    exit(0)

def train_hmm(digit):

    data0 = np.loadtxt(digit+'_0.mfcc')
    data1 = np.loadtxt(digit+'_1.mfcc')
    data2 = np.loadtxt(digit+'_2.mfcc')
    data3 = np.loadtxt(digit+'_3.mfcc')
    data4 = np.loadtxt(digit+'_4.mfcc')

    segs = np.ones((5,4)) * np.array([0.2,0.4,0.6,0.8])
    segs = np.array([[data0.shape[0]],
                     [data1.shape[0]],
                     [data2.shape[0]],
                     [data3.shape[0]],
                     [data4.shape[0]]]) * segs

    # HMM: Our HMM will be a numpy matrix 10 x 39
    # because 5 states and one row for mean and one row for variance
    HMM = np.zeros((10,data0.shape[1]))

    # Transition probabilities
    trans_mat = np.array([[0.000001, 0.0000001, 1.0],
                          [0.000001, 0.5, 0.5],
                          [0.7, 0.15, 0.15],
                          [0.7, 0.15, 0.15],
                          [0.7, 0.15, 0.15],
                          [0.5, 0.2, 0],
                          [0.3, 0, 0]])

    # Convert to distance
    trans_mat = - np.log(trans_mat) # Will give warnings
    
         
    # Extract appropriate sections for each state
    state1 = np.concatenate((data0[:segs[0][0],:],
                             data1[:segs[1][0],:],
                             data2[:segs[2][0],:],
                             data3[:segs[3][0],:],
                             data4[:segs[4][0],:]),axis=0)

    HMM[0][:] = np.mean(state1,axis=0)
    HMM[1][:] = np.diag(np.cov(state1, rowvar=0))



    state1 = np.concatenate((data0[segs[0][0]:segs[0][1],:],
                             data1[segs[1][0]:segs[1][1],:],
                             data2[segs[2][0]:segs[2][1],:],
                             data3[segs[3][0]:segs[3][1],:],
                             data4[segs[4][0]:segs[4][1],:]),axis=0)

    HMM[2][:] = np.mean(state1,axis=0)
    HMM[3][:] = np.diag(np.cov(state1, rowvar=0))
    


    state1 = np.concatenate((data0[segs[0][1]:segs[0][2],:],
                             data1[segs[1][1]:segs[1][2],:],
                             data2[segs[2][1]:segs[2][2],:],
                             data3[segs[3][1]:segs[3][2],:],
                             data4[segs[4][1]:segs[4][2],:]),axis=0)

    HMM[4][:] = np.mean(state1,axis=0)
    HMM[5][:] = np.diag(np.cov(state1, rowvar=0))



    state1 = np.concatenate((data0[segs[0][2]:segs[0][3],:],
                             data1[segs[1][2]:segs[1][3],:],
                             data2[segs[2][2]:segs[2][3],:],
                             data3[segs[3][2]:segs[3][3],:],
                             data4[segs[4][2]:segs[4][3],:]),axis=0)

    HMM[6][:] = np.mean(state1,axis=0)
    HMM[7][:] = np.diag(np.cov(state1, rowvar=0))
    


    state1 = np.concatenate((data0[segs[0][3]:,:],
                             data1[segs[1][3]:,:],
                             data2[segs[2][3]:,:],
                             data3[segs[3][3]:,:],
                             data4[segs[4][3]:,:]),axis=0)

    HMM[8][:] = np.mean(state1,axis=0)
    HMM[9][:] = np.diag(np.cov(state1, rowvar=0))

    best_overall_cost = np.inf
    best_overall_segs = None
    best_overall_trans = None

    for i in xrange(60):

        # Do DTW between HMM and data sequence
        new_segs0, new_tr0, best_cost0 = do_DTW(HMM,trans_mat,data0)
        new_segs1, new_tr1, best_cost1 = do_DTW(HMM,trans_mat,data1)
        new_segs2, new_tr2, best_cost2 = do_DTW(HMM,trans_mat,data2)
        new_segs3, new_tr3, best_cost3 = do_DTW(HMM,trans_mat,data3)
        new_segs4, new_tr4, best_cost4 = do_DTW(HMM,trans_mat,data4)

        avg_best_cost = 0.25 * (best_cost0 + best_cost1 + best_cost2
                                + best_cost3 + best_cost4)

        print 'Iteration',i,' cost: ',avg_best_cost
        # Update rules
    
        segs = np.concatenate((new_segs0.transpose(),
                               new_segs1.transpose(),
                               new_segs2.transpose(),
                               new_segs3.transpose(),
                               new_segs4.transpose()), axis=0)

        trans_mat[2:,:] = trans_mat[2:,:] + 0.025* (new_tr0 + new_tr1 + new_tr2
                                                    + new_tr3 + new_tr4)
        
        # Extract appropriate sections for each state
        state1 = np.concatenate((data0[:segs[0][0]+1,:],
                                 data1[:segs[1][0]+1,:],
                                 data2[:segs[2][0]+1,:],
                                 data3[:segs[3][0]+1,:],
                                 data4[:segs[4][0]+1,:]),axis=0)

        HMM[0][:] = np.mean(state1,axis=0)
        HMM[1][:] = np.diag(np.cov(state1, rowvar=0))



        state1 = np.concatenate((data0[segs[0][0]-1:segs[0][1]+1,:],
                                 data1[segs[1][0]-1:segs[1][1]+1,:],
                                 data2[segs[2][0]-1:segs[2][1]+1,:],
                                 data3[segs[3][0]-1:segs[3][1]+1,:],
                                 data4[segs[4][0]-1:segs[4][1]+1,:]),axis=0)

        HMM[2][:] = np.mean(state1,axis=0)
        HMM[3][:] = np.diag(np.cov(state1, rowvar=0))
    


        state1 = np.concatenate((data0[segs[0][1]-1:segs[0][2]+1,:],
                                 data1[segs[1][1]-1:segs[1][2]+1,:],
                                 data2[segs[2][1]-1:segs[2][2]+1,:],
                                 data3[segs[3][1]-1:segs[3][2]+1,:],
                                 data4[segs[4][1]-1:segs[4][2]+1,:]),axis=0)

        HMM[4][:] = np.mean(state1,axis=0)
        HMM[5][:] = np.diag(np.cov(state1, rowvar=0))



        state1 = np.concatenate((data0[segs[0][2]-1:segs[0][3]+1,:],
                                 data1[segs[1][2]-1:segs[1][3]+1,:],
                                 data2[segs[2][2]-1:segs[2][3]+1,:],
                                 data3[segs[3][2]-1:segs[3][3]+1,:],
                                 data4[segs[4][2]-1:segs[4][3]+1,:]),axis=0)

        HMM[6][:] = np.mean(state1,axis=0)
        HMM[7][:] = np.diag(np.cov(state1, rowvar=0))
    


        state1 = np.concatenate((data0[segs[0][3]-1:,:],
                                 data1[segs[1][3]-1:,:],
                                 data2[segs[2][3]-1:,:],
                                 data3[segs[3][3]-1:,:],
                                 data4[segs[4][3]-1:,:]),axis=0)

        HMM[8][:] = np.mean(state1,axis=0)
        HMM[9][:] = np.diag(np.cov(state1, rowvar=0))

        
    filnm = digit+'.hmm'
    np.savetxt(filnm,HMM)
    filnm = digit+'.trans'
    np.savetxt(filnm,trans_mat)

    pass

if __name__ == '__main__':
    train_hmm(sys.argv[1])
