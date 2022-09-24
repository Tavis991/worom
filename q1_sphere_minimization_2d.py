import itertools as itr
import numpy as np 
import matplotlib.pyplot as plt 

MIU_SIZE = 25
ITER_COUNT = 100
EPS = 0.003
U_BOUND = 2
L_BOUND = -1
DIM = 80

class hypervol_solver():
    def __init__(self, funcs):
        self.counter = 0
        if len(funcs) > 3 or len(funcs) < 1 : 
            raise Exception('1-3 objective functions per solver')
        self.funcs = funcs
        self.x_vec = np.random.uniform(L_BOUND, U_BOUND, (500, DIM))# $
        self.y_vec = self.evaluation(self.x_vec)

    def evaluation (self, x_vec):
        y_vec = np.array([[self.funcs[i](x_vec[j]) for j in range \
             (x_vec.shape[0])] for i in range (len(self.funcs))] ).T
        return y_vec

    def build_front(self):
        y = self.is_pareto_efficient_simple(self.y_vec)
        
        self.front_size = sum(y)
        miu_y = self.y_vec[y]  
        miu_x = self.x_vec[y]

        self.front = [np.zeros((miu_x.shape)), np.zeros((miu_y.shape))]
        self.front[0] = np.copy(miu_x)
        self.front[1] = np.copy(miu_y)
        
    def iterate(self): 
        while self.front[0].shape[0] < 100 :
            for pair in list (itr.combinations(self.front[0], 2)) : #to prevent population extinction 
                self.front[0] = np.append(self.front[0], self.recombine(pair[0],pair[1]).reshape(1,80), axis=0)

        self.y_vec = self.evaluation(self.front[0])
        self.x_vec = self.front[0] 
        self.build_front()   

    def solve(self):
        self.build_front()
   
        diff = np.inf
        vol = 0
        vols = []
        while(self.counter < ITER_COUNT):
            self.counter += 1
            self.iterate() 

            vol_n = self.hypervol() #remove least contributors and return volume sum
            vols.append(vol_n)
            diff = vol_n - vol  #add new point, must remain undominated set
            
            vol = vol_n 
        fig = plt.figure()
        ax1 = fig.add_subplot(111)        
        ax1.scatter(self.y_vec[:,0], self.y_vec[:,1], color='blue')
        ax1.scatter(self.front[1][:,0], self.front[1][:,1], color = 'yellow')
        plt.pause(1)
        plt.show()
        last_vol = self.hypervol()
        return self.front

    def hypervol (self):
        reference_point = self.init_dystopia()  #ref point is the worse x,y,(z) of all miu 

        dim = reference_point.shape[0]#volume to be filled 
      
        if dim == 2 :
            while self.front_size > MIU_SIZE:
                front_by_y = np.copy(self.front[1])
                idexs = front_by_y[:, 1].argsort()
                vols_by_y = front_by_y[idexs]
                differences = np.diff(vols_by_y,axis = 0) 
                contributions = np.abs(differences[:-1, 0] * differences[1:, 1])
                self.front[1] = np.delete(self.front[1], idexs[np.argmin(contributions)+1], axis=0)
                self.front[0] = np.delete(self.front[0], idexs[np.argmin(contributions)+1], axis=0)
                self.front_size -= 1
            #updating volume without lsp 
            front_y_2 = np.copy(self.front[1])
            idexs_2 = front_y_2[:, 1].argsort()
            vols_y_2 = front_y_2[idexs_2]
            differences_2 = np.diff(vols_y_2,axis = 0)
            
                        
        
        
            contributions_2 = np.abs(differences_2[:-1, 0] * differences_2[1:, 1]) 

        return sum(contributions_2)

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
        new_p[mut] = np.random.normal(np.mean(p1), 0.2) #to do more mutations, good for ones together
        return new_p

    def init_dystopia(self) : #any dimensional
        miu = np.copy(self.front[1])
        dim = len(self.funcs) 

        dystopia = np.full(dim,np.inf)
        for m in range(dim) :
            dystopia[m] = np.max(miu[:,m])
        return dystopia 

    def is_pareto_efficient_simple(self, costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        return is_efficient

    def best_point(self):
        y = self.front[1][np.argmin(np.linalg.norm(self.front[1], axis=1))]
        x = self.front[0][np.argmin(np.linalg.norm(self.front[1], axis=1))]
        p = np.min(np.linalg.norm(self.front[1], axis=1))
        return p,y,x

# solve for n = 80
f1_1 = lambda x1 : np.dot(x1.T , x1) # minimize
f1_2 = lambda x1 : np.dot((x1-1).T, x1-1) # minimize


funcs = [f1_1,f1_2]
solver = hypervol_solver(funcs)
front = solver.solve()

print(front[1][np.argmin(np.linalg.norm(front[1], axis=1))])
print(front)
x = front[0][np.argmin(np.linalg.norm(front[1], axis=1))]