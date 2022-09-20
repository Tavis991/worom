# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:14:55 2022

@author: Eden Akiva
"""

import itertools as itr
#import pandas as pd 
import numpy as np 
#import os
import matplotlib.pyplot as plt 
#from scipy.spatial import ConvexHull
import pandas as pd

MIU_SIZE = 25
ITER_COUNT = 100
EPS = 0.003
U_BOUND = 100
L_BOUND = 0
DIM = 20

class hypervol_solver():
    def __init__(self, funcs, covariance, wealth, returns):
        self.covariance = np.array(covariance)
        self.returns = np.array(returns)
        self.wealth = wealth
        self.counter = 0
        if len(funcs) > 3 or len(funcs) < 1 : 
            raise Exception('1-3 objective functions per solver')
        self.funcs = funcs
        k = np.random.rand(500,20) #20 investment
        k3 = self.normalize(k) # normalizing 
        self.x_vec = k3
        # for i in range (500) : 
        #     self.x_vec[i][np.random.randint(0,19)] += (100 - sum(k3)[i]) # remainder 
        self.y_vec = self.evaluation(self.x_vec)

        # evaluated_f1_sp = self.funcs[0](samples[:,0],samples[:,1]) #
        # for f in funcs : 
            
        #     evaluated_f2_sp = self.funcs[1](samples[:,0],samples[:,1])  # implementation specific 

        # res = np.stack((evaluated_f1_sp, evaluated_f2_sp), axis=1)
    
    def evaluation (self, x_vec):
        info = [self.covariance, self.returns] 
        y_vec = np.array([[self.funcs[i](x_vec[j], info[i]) for j in range \
             (x_vec.shape[0])] for i in range (len(self.funcs))] ).T
        return y_vec

    def build_front(self):
        y = self.pareto_ranking(self.y_vec)
        self.front_size = len(y) - np.count_nonzero(y)   
        arg_v = np.argsort(y)[:self.front_size] #min(front_size, MIU_SIZE)

        miu_y = self.y_vec[arg_v]  
        miu_x = self.x_vec[arg_v]
        
        # MIU_SIZE #to prevent population explosion  #DOES NOT TAKE INTO ACCCOUNT COVERAGE AREA HERE
        # if front_size > MIU_SIZE:
        #     sorted_miu = np.linalg.norm(miu_y, axis=1)
        #     min_inde = np.argsort(sorted_miu)[:MIU_SIZE] 
        #     miu_y = miu_y[min_inde]
        #     miu_x = miu_x[min_inde]

        self.front = [np.zeros((miu_x.shape)), np.zeros((miu_y.shape))]
        self.front[0] = np.copy(miu_x)
        self.front[1] = np.copy(miu_y)
    
    def normalize(self, vec):
        zi = (vec / (np.sum (vec, axis=1)).reshape(len(vec), 1)) * self.wealth
        for i in range (len(zi)) : 
           zi[i][np.random.randint(0, len(returns)-1)] += (100 - np.sum(zi, axis=1)[i])
        return zi 

    def iterate(self): 
        while self.front[0].shape[0] < 100 :
            for pair in list (itr.combinations(self.front[0], 2)) : #to prevent population extinction 
                self.front[0] = np.append(self.front[0], self.normalize(self.recombine(pair[0],pair[1]).reshape(1,20)), axis=0)

        self.y_vec = self.evaluation(self.front[0])
        self.x_vec = self.front[0]
        self.build_front()   

    def solve(self):
        self.build_front()
   
        diff = np.inf
        vol = 0
        vols = []
        while(self.counter < ITER_COUNT): # (diff > EPS) and  #stop conditions TODO change 
            self.counter += 1
            vol_n = self.hypervol() #remove least contributor
            vols.append(vol_n)
            diff = vol_n - vol  #add new point, must remain undominated set
            
            vol = vol_n 
            self.iterate()
        return self.front

    def hypervol (self):
        reference_point = self.init_dystopia()  #ref point is the worse x,y,(z) of all miu 
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)        
        ax1.scatter(self.y_vec[:,0], self.y_vec[:,1], color='blue')
        ax1.scatter(self.front[1][:,0], self.front[1][:,1], color = 'yellow')
        ax1.scatter(reference_point[0], reference_point[1], color = 'red')
        plt.pause(1)
        plt.show()
        
        dim = reference_point.shape[0]#volume to be filled 
        #front_by_y = np.copy(self.front[1])
      
        if dim == 2 :
            while self.front_size > MIU_SIZE:
                front_by_y = np.copy(self.front[1])
                idexs = front_by_y[:, 1].argsort()
                vols_by_y = front_by_y[idexs]
                differences = np.diff(vols_by_y,axis = 0) #CAN GET RID OF EDGE HERE I THINK
                contributions = np.abs(differences[:-1, 0] * differences[1:, 1]) #ןIS THIS RIGHT??
                self.front[1] = np.delete(self.front[1], idexs[np.argmin(contributions)+1], axis=0)
                self.front[0] = np.delete(self.front[0], idexs[np.argmin(contributions)+1], axis=0)
                self.front_size -= 1
            #updating volume without lsp # technically only need to update two, but..
            front_y_2 = np.copy(self.front[1])
            idexs_2 = front_y_2[:, 1].argsort()
            vols_y_2 = front_y_2[idexs_2]
            differences_2 = np.diff(vols_y_2,axis = 0)
            
            contributions_2 = np.abs(differences_2[:-1, 0] * differences_2[1:, 1]) #NEED TO UPDATE DIFFFERNCES FOR THIS TO DO SOMETHING

        return sum(contributions_2)

    def inclusive(self, refpoint, point): #what does this do??
        """
        return the product of the difference between all objectives
        and a reference point
        """
        offset = [p-r for (p,r) in zip(point, refpoint)]
        volume = 1
        for val in offset:
            volume *= val
        return abs(volume)

    def recombine(self, p1,p2): 
        offset = np.random.randint(1,p1.shape[0])
        mut = np.random.randint(0,p1.shape[0])

        new_p = np.concatenate((p1[:offset],p2[offset:]),axis=0)
        #new_p[mut] = new_p[mut] * (1 + np.random.normal() * 0.01) #TODO boundry problems 
        new_p[mut] = np.random.normal(np.mean(p1), 0.2)
        return new_p

    def init_dystopia(self) : #any dimensional
        miu = np.copy(self.front[1])
        dim = len(self.funcs) # miu[0].shape[0]  #WHY NOT JUST 

        dystopia = np.full(dim,np.inf)
        for m in range(dim) :
            dystopia[m] = np.max(miu[:,m]) #CHANGED MIU TO Y_VEC
        return dystopia #DO I NEED TO MAKE A COPY OF Y_VEC?
    def dominate(self, x1: np.array, x2: np.array):
        return (x1 <= x2).all() and (x1 < x2).any()

    def pareto_ranking(self, ObjFncVectors):
        X = pd.DataFrame(ObjFncVectors)
        return np.array([np.sum(X.T.apply(lambda x: self.dominate(x, X.T[i]), axis=0)) for i in range(X.shape[0])])
    # def pareto_ranking(self, ObjFncVectors) :  #any dimensional
    #     ranking = np.zeros(len(ObjFncVectors)) #WHY NOT JUST SELF.Y_VEC.shape[0]?? ""^

    #     for idx in range(len(ObjFncVectors)): #dominating vec 
    #         for idx2 in range(len(ObjFncVectors)): #dominated vec 
    #             if np.all(ObjFncVectors[idx, :] <= ObjFncVectors[idx2, :]) \
    #                 and np.any(ObjFncVectors[idx, :] < ObjFncVectors[idx2, :] ) :
    #                 ranking[idx2] += 1 
    #     return ranking

    def best_point(self):
        y = self.front[1][np.argmin(np.linalg.norm(self.front[1], axis=1))]
        x = self.front[0][np.argmin(np.linalg.norm(self.front[1], axis=1))]
        p = np.min(np.linalg.norm(self.front[1], axis=1))
        return p,y,x

# solve for n = 80
f1_1 = lambda x1 : np.dot(x1.T , x1) # minimize
f1_2 = lambda x1 : np.dot((x1-1).T, x1-1) # minimize

# Q = covariance matrix nxn of portfolio investments       #given as input
# rho = expected return on investment vect n               #given as input
f2_risk = lambda x, Q : np.dot( np.dot(x.T,Q) , x) # minimize
neg_f2_return = lambda x, rho: - (np.dot(x.T , rho)) #MAXIMIZE- minimize neg
f2_B = lambda x : np.sum(x)

# solve for n = 30
f3_1 = lambda x1 : np.dot(x1.T , x1) # minimize
f3_2 = lambda x1 : np.dot((x1-1).T, x1-1) # minimize
f3_3 = lambda x1 : np.dot((x1-2).T, x1-2) # minimize

cov_parse = lambda row : [float(i) for i in row[2:-3].split(',')]

df = pd.read_table('portfolio.dat') # ['Investments' , 'Return' , 'Covariance' , 'Wealth']
returns = [float(i) for i in df.values[7][0].split('=')[1].split(';')[0][2:-2].split(',')] 
covariance = [cov_parse(df.values[9+i][0]) for i in range(20)]
wealth = int(df.values[30][0].split('=')[1].split(';')[0])
funcs = [f2_risk, neg_f2_return]

solver = hypervol_solver(funcs, covariance, wealth, returns)
front = solver.solve()


print(front[1][np.argmin(np.linalg.norm(front[1], axis=1))])

x = front[0][np.argmin(np.linalg.norm(front[1], axis=1))]


# diff = solver.y_vec



# while diff < EPS : 
#     #add new point  
#     evaluated_f1_sp = f1(miu[:,0],samples[:,1])  
#     evaluated_f2_sp = f2(samples[:,0],samples[:,1])
#     res = np.stack((evaluated_f1_sp, evaluated_f2_sp), axis=1)
#             #TODO recombination + selection 
#     y = pareto_ranking(res)
#     arg_v = np.argsort(y)[:MIU_SIZE]
#     miu = res[arg_v]
#     reference_point = init_dystopia(res[arg_v])
#     front_size = len(samples) -  np.count_nonzero(y)  
#     for q in front_size :  #calculte vol with and withotu each point 
#         vol = hypervol(miu, front_size)

# plt.scatter(evaluated_f1_sp[y > 0], evaluated_f2_sp[y > 0])  
# plt.scatter(evaluated_f1_sp[y == 0], evaluated_f2_sp[y == 0], color='green')
# plt.show()


# path = os.path.join(os.getcwd(), 'DMA_q1.dat')
# file = open(path ,"w")
# file.write(str([list(smp) for smp in samples]))
# file.close()
