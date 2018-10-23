# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:06:39 2018

@author: maisuiY
@algorithm:NSGA-2
@environment:windows 10 anaconda spyder
"""
import numpy as np


def f_x1(x1):
    return x1*x1


def f_x2(x2):
    return (x2-2)*(x2-2)


def fast_non_dominated_sort(P, N, l, rank_p,length_Fy,R_decode):   # N records half of the all population of merged_Population R
    S_p = np.zeros([2*N,2*N])-1 # each column represents a initial Sp (p=0,...,399), which includes 400 units at most to save the location of q that p dominates.
    S_p = S_p.astype(int)
    length_Sp=np.zeros( 2*N )
    length_Sp=length_Sp.astype(int)
    F_y=np.zeros([2*N,2*N])-1   #each column represents a initial set Fy (y=0,...,399(at most)), which includes 400 units at most to save the location of y that belong to the same rank.
    F_y=F_y.astype(int)
    Number_p=np.zeros( 2*N )
    k=0
    for i in range(2*N):
        Number_p[i]=0
        j=0
        while(j<2*N):   # whether to keep i is equal to j or not
            if(  (f_x1(R_decode[i]) < f_x1(R_decode[j]) and f_x2(R_decode[i]) < f_x2(R_decode[j]) ) or
                 (f_x1(R_decode[i]) < f_x1(R_decode[j])and f_x2(R_decode[i]) <= f_x2(R_decode[j]) ) or
                 (f_x1(R_decode[i]) <= f_x1(R_decode[j]) and f_x2(R_decode[i]) < f_x2(R_decode[j]) )  ):
                S_p[ int(length_Sp[i]),i ]=j
                length_Sp[i]+=1
            elif( (f_x1(R_decode[i]) > f_x1(R_decode[j]) and f_x2(R_decode[i]) > f_x2(R_decode[j]) ) or
                 (f_x1(R_decode[i]) > f_x1(R_decode[j])and f_x2(R_decode[i]) >= f_x2(R_decode[j]) ) or
                 (f_x1(R_decode[i]) >= f_x1(R_decode[j]) and f_x2(R_decode[i]) > f_x2(R_decode[j]) )  ):
                Number_p[i]+=1
            j+=1
        if(np.abs(Number_p[i])<0.01):
            rank_p[i]=0
            F_y[k,0]=i
            k+=1
    length_Fy[0] = k
    i = 0
    while(np.sum(F_y[:,i])  >-400 ):
        Q=np.zeros(2*N)-1
        k=0  #clear k for next using
        for p in F_y[:,i]:
            if(p>=0):
                for q in S_p[:,p]:
                    if(q>=0):
                        Number_p[q] -= 1
                        if( np.abs(Number_p[q])<0.01):
                            rank_p[ q ]=i+1
                            Q[k]=q
                            k+=1
        i+=1
        length_Fy[i]=k
        F_y[:,i]=Q
    return F_y
def crowd_distance_assignment( L,num_solu,R_decode):
    distance_L=np.zeros([num_solu,3])  # the column 0 record the L( record the location of solution ),the column 1 record the distance of each solution
    distance_L=np.reshape(distance_L,[num_solu,3])
    distance_L[:,0]=L[0:num_solu]
    distance_L[:,1]=f_x1(R_decode[ L[0:num_solu] ].reshape(num_solu))
    distance_L[:,2]=f_x2(R_decode[ L[0:num_solu] ].reshape(num_solu))
    for i in range(1,3):
        f_max=np.max(distance_L[:,i])
        f_min=np.min(distance_L[:,i])
        distance_L = distance_L[distance_L[:,i].argsort()]
        distance_L[0,i]=99999999
        distance_L[num_solu-1,i]=99999999
        for j in range(num_solu-3):
            distance_L[j+1,i]=(distance_L[j+2,i]-distance_L[j,i])/(f_max-f_min)
    distance_final=np.zeros([num_solu,2])
    distance_final=np.reshape(distance_final,[num_solu,2])
    distance_final[:,0]=distance_L[:,0]
    distance_final[:,1]=np.sum( distance_L,1)-distance_L[:,0]
    distance_final =distance_final[distance_final[:,1].argsort()] #sort the set L by the crowded_comparison operator<ascending>
    return distance_final
def make_new_pop(Parent):
    Offspring=np.array(Parent)
    Offspring=Offspring.reshape(N,l)
    Prob_cross=0.9 # the crossover probablity
    Prob_muta=1/l # the mutation probablity
    cross_or_not=np.random.rand()
    if cross_or_not<Prob_cross:                # crossover
        assist1=np.arange(N)
        np.random.shuffle(assist1)
        temp1=np.random.randint(0,l)
        for i in range(int(N/2)):
            r=np.array( Offspring[ assist1[2*i],temp1:l ])
            Offspring[assist1[2*i],temp1:l]=Offspring[assist1[2*i+1],temp1:l]
            Offspring[assist1[2*i+1],temp1:l]=r
    for i in range(N):   # mutation
        muta_ornot=np.random.rand()
        if muta_ornot<Prob_muta:
            temp2=np.random.randint(0,l)
            Offspring[i,temp2]=1-Offspring[i,temp2]
    return Offspring


N=200  # the number of population in parent
l=30 # the number of bits when binary-coding
generation=10
Parent=np.random.randint(0,2,size=[N,l])
Parent=Parent.reshape(N,l)
Offspring=make_new_pop(Parent)
for genera in range(generation):
    R=np.vstack((Parent,Offspring))
    R=R.reshape(2*N,l)
    rank_p=np.zeros( 2*N )
    length_Fy=np.zeros([ 2*N ],int)
    R_decode=np.zeros(2*N)
    R_decode=R_decode.reshape(2*N,1)
    for i in range( l ):
        R_decode=R[:,i].reshape(2*N,1)*np.power(2,i)+R_decode         #remind to clear the variables
    R_decode=-1000+R_decode*2000/(np.power(2,30)-1)
    F=fast_non_dominated_sort(R,N,l,rank_p,length_Fy,R_decode)
    sum_first2i=0
    for i in range(2*N):  # the flag to record the location of F_y
        sum_first2i=length_Fy[i]+sum_first2i
        if(sum_first2i>N):
            break
    guide_Parent_new=np.zeros([N,1],int)
    np.reshape(guide_Parent_new,[N,1])
    j=0  # j is defined to visit the matrix F
    k=0  # k is defined to visit the guide_Parent_new
    while(j<i):
        guide_Parent_new[k:k+length_Fy[j]]=np.reshape( F[0:length_Fy[j],j],[length_Fy[j],1])
        k+=length_Fy[j]  # k must larger or equal to the N
        j+=1
    if( k<N ):
        Len=crowd_distance_assignment( F[:,j],length_Fy[j],R_decode )
        guide_Parent_new[k:N-1]=Len[length_Fy[j]-(N-k) ,0]
    guide_Parent_new=guide_Parent_new.reshape(200)
    Parent_new=R[guide_Parent_new,:]
    Parent=Parent_new
    Offspring=make_new_pop(Parent_new)
    print("the generation is %d\n th",genera )











