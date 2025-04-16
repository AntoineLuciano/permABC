from dataclasses import dataclass
import jax.numpy as jnp
import particles
import numpy as np
from jax import random

@dataclass
class Theta:
    loc: np.ndarray
    glob: np.ndarray
    
    def __init__(self, loc= np.array([]), glob=np.array([])):
        self.loc = np.array(loc)
        self.glob = np.array(glob)
      
    def apply_permutation(self, index):
        # if self.loc.shape[1]==1:
        #     print("PROBLEME SHAPE THETAS LOC")
        #     return self
        return Theta(loc =self.loc[np.arange(self.loc.shape[0])[:, None], index], glob= self.glob)
    
    def __getitem__(self, index):
        return Theta(loc = self.loc[index], glob = self.glob[index])
    
    def __setitem__(self, index, value):
        self.loc[index] = value.loc
        self.glob[index] = value.glob
    
    
    def reshape_2d(self):
        n_particles = self.loc.shape[0]
        return np.concatenate((self.loc.reshape(n_particles,-1),self.glob), axis = 1).reshape(n_particles,-1)
    
    def copy(self):
        return Theta(loc = self.loc.copy(), glob = self.glob.copy())
    
    def shape(self):
        return self.loc.shape, self.glob.shape
    
    def append(self, other):
        if len(self)==0:
            self.loc = other.loc
            self.glob = other.glob
        else:
            self.loc = np.concatenate((self.loc, other.loc), axis = 0)
            self.glob = np.concatenate((self.glob, other.glob), axis = 0)
        
    def __len__(self):
        return self.loc.shape[0]
    
    def truncating(self, new_M, old_M):
        self.loc = np.concatenate((self.loc[:,:new_M], self.loc[:,old_M:]), axis = 1)
        
    
    def __eq__(self, other):
        return np.all(self.loc == other.loc) and np.all(self.glob == other.glob)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def duplicate(self, n_duplicate, perm_duplicated):
        if perm_duplicated.shape[0]%n_duplicate != 0:
            raise ValueError("The number of duplicates should be a multiple of the number of particles")
        new_loc = np.repeat(self.loc, n_duplicate, axis = 0)
        new_glob = np.repeat(self.glob, n_duplicate, axis = 0)
        new_thetas = Theta(loc = new_loc, glob = new_glob)
        new_thetas = new_thetas.apply_permutation(perm_duplicated)
        return new_thetas
        
    
        


def resampling(key, weight, L_to_resample, n_particles = 0):
    if n_particles==0: n_particles = len(weight)
    index = particles.resampling.systematic(weight,n_particles)
    # index = random.choice(key, np.where(weight>0)[0], shape = (n_particles,), replace = True)
    return [to_resample[index] for to_resample in L_to_resample]

def ess(weight):
    return np.round(1/sum(weight**2))


