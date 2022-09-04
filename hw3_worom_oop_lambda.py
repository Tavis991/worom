import itertools as itr
#import pandas as pd 
import numpy as np 
#import os
import matplotlib.pyplot as plt 
#from scipy.spatial import ConvexHull

MIU_SIZE = 50 
EPS = 0.003
U_BOUND = 5
L_BOUND = -5
DIM = 80

class hypervol_solver():
    def __init__(self, funcs):
        if len(funcs) > 3 or len(funcs) < 1 : 
            raise Exception('1-3 objective functions per solver')
        self.funcs = funcs
        self.x_vec = np.random.uniform(L_BOUND, U_BOUND, (500, DIM))# $
        self.y_vec = self.evaluation(self.x_vec)
        # evaluated_f1_sp = self.funcs[0](samples[:,0],samples[:,1]) #
        # for f in funcs : 
            
        #     evaluated_f2_sp = self.funcs[1](samples[:,0],samples[:,1])  # implementation specific 

        # res = np.stack((evaluated_f1_sp, evaluated_f2_sp), axis=1)
    
    def evaluation (self, x_vec):
        #x_vec = np.array([x_vec[y] for y in range (DIM)])
        y_vec = np.array([[self.funcs[i](x_vec[j]) for j in range \
             (x_vec.shape[0])] for i in range (len(self.funcs))] ).T
        return y_vec

    def build_front(self):
        y = self.pareto_ranking(self.y_vec)
        arg_v = np.argsort(y)[:MIU_SIZE]
        miu_y = self.y_vec[arg_v]  
        miu_x = self.x_vec[arg_v]

        self.front = [np.zeros((miu_x.shape)), np.zeros((miu_y.shape))]
        self.front[0] = np.copy(miu_x)
        self.front[1] = np.copy(miu_y)
        
    def iterate(self): 
        for pair in list (itr.combinations(self.front[0], 2)) :
            np.append(self.front, self.recombine(pair[0],pair[1]))
            self.y_vec = self.evaluation(self.front[0])
            self.x_vec = self.front[0]
            self.build_front()

    def solve(self):
        self.build_front()

        # while (len(self.front) < MIU_SIZE) : #we should remain only with front points 
        #     self.iterate()
        diff = np.inf
        vol = 0
        while (diff > EPS):
            vol_n = self.hypervol() #remove least contributor
            diff = vol_n - vol  #add new point, must remain undominated set
            vol = vol_n 

    def hypervol (self):
        reference_point = self.init_dystopia()  #ref point is the worse x,y,(z) of all miu 
        dim = reference_point.shape[0]#volume to be filled 
        no_border = np.copy(self.front[1])

        for m in range (dim):
            no_border = np.delete(no_border, np.argmax(self.front[1][:,m]), axis=0) #removing from group the border points, contribute 0 volume 
        
        vols_inclus = []
        for p in no_border :
            vols_inclus.append(self.inclusive(reference_point, p))
        return (sum(vols_inclus))    
        #TODO how to detecet overlap 

    def inclusive(self, refpoint, point):
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
        new_p[mut] = new_p[mut] * (1 + np.random.normal() * 0.01) #TODO boundry problems 
        return new_p

    def init_dystopia(self) : #any dimensional
        miu = np.copy(self.front[1])
        dim = miu[0].shape[0]
        dystopia = np.full(dim,np.inf)
        for m in range(dim) :
            dystopia[m] = np.max(miu[:,m]) 
        return dystopia

    def pareto_ranking(self, ObjFncVectors) :  #any dimensional
        ranking = np.zeros(len(ObjFncVectors))
        for idx in range(len(ObjFncVectors)): #dominating vec 
            for idx2 in range(len(ObjFncVectors)): #dominated vec 
                if np.all(ObjFncVectors[idx,:] <= ObjFncVectors[idx2,:]) \
                    and np.any(ObjFncVectors[idx,:] < ObjFncVectors[idx2,:] ) :
                    ranking[idx2] += 1 
        return ranking



    
f1 = lambda x1,x2 : x1 ** 2 + (x2 - 0.5) ** 2#$
f2 = lambda x1,x2 : (x1 - 1) ** 2 + (x2 - 0.5) ** 2#$ implementation specific 



# solve for n = 80
f1_1 = lambda x1 : np.dot(x1.T , x1) # minimize
f1_2 = lambda x1 : np.dot((x1-1).T, x1-1) # minimize

# Q = np.matrix #covariance matrix nxn of portfolio investments       #given as input
# rho = np.vector #expected return on investment vect n               #given as input
# f2_risk = lambda x, Q : np.dot( np.dot(x.T,Q) , x) # minimize
# f2_return = lambda x, rho: np.dot(x.T , rho) #MAXIMIZE
# f2_B = lambda x : np.sum(x)

# solve for n = 30
f3_1 = lambda x1 : np.dot(x1.T , x1) # minimize
f3_2 = lambda x1 : np.dot((x1-1).T, x1-1) # minimize
f3_3 = lambda x1 : np.dot((x1-2).T, x1-2) # minimize

funcs = [f1_1,f1_2]
solver = hypervol_solver(funcs)
solver.solve()
diff = solver.y_vec


while diff < EPS : 
    #add new point  
    evaluated_f1_sp = f1(miu[:,0],samples[:,1])  
    evaluated_f2_sp = f2(samples[:,0],samples[:,1])
    res = np.stack((evaluated_f1_sp, evaluated_f2_sp), axis=1)
            #TODO recombination + selection 
    y = pareto_ranking(res)
    arg_v = np.argsort(y)[:MIU_SIZE]
    miu = res[arg_v]
    reference_point = init_dystopia(res[arg_v])
    front_size = len(samples) -  np.count_nonzero(y)  
    for q in front_size :  #calculte vol with and withotu each point 
        vol = hypervol(miu, front_size)

plt.scatter(evaluated_f1_sp[y > 0], evaluated_f2_sp[y > 0])  
plt.scatter(evaluated_f1_sp[y == 0], evaluated_f2_sp[y == 0], color='green')
plt.show()


# path = os.path.join(os.getcwd(), 'DMA_q1.dat')
# file = open(path ,"w")
# file.write(str([list(smp) for smp in samples]))
# file.close()