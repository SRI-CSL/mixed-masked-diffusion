import jax.numpy as jnp
import numpy as np
from copy import deepcopy
import jax

class dataset:
    """Data container with normalization, denormalization, splitting, selection, and sampling utilities.

    Handles NaN/inf, tpology mask storage and calculation, and provides methods for drawing
    random samples.
    """

    def __init__(self, datain, mean = None, stdev = None, norm = False, labels = None, limits = None, iblank = None):
        datain = jnp.array(datain)

        # Remove all inf values
        datain = datain[(~jnp.isinf(datain).any(axis=1)[:, 0]), :, :]

        # Initialize iblank as None
        if iblank is None:
            self.iblank = datain.shape[1]
        else:
            self.iblank = iblank

        if mean is None:
            self.mean = jnp.nanmean(datain,axis=0,keepdims=True)
        else:
            self.mean = mean
        if stdev is None:
            self.stdev = jnp.nanstd(datain,axis=0,keepdims=True)
            self.stdev = jnp.where(self.stdev==0., 1, self.stdev)
        else:
            self.stdev = stdev
        self.nodes_max = datain.shape[1]
        self.node_ids = jnp.arange(self.nodes_max)

        if norm == False:
            self.data = datain
            self.normalize()

        elif norm == True:
            self.norm = datain
            self.unnormalize()

        if labels is None:
            labels= list(map(chr, range(97, 97+self.nodes_max)))
        self.labels = np.array(labels)

        if limits is None:
            max = jnp.nanmax(datain,axis=0)
            min = jnp.nanmin(datain,axis=0)
            self.limits = jnp.hstack((min,max))
        else:
            self.limits = jnp.array(limits)
            max = self.limits[:,1,None]
            min = self.limits[:,0,None]

        removal = jnp.argwhere(max[:,0]==min[:,0]).squeeze()
        self.removed = removal


    def normalize(self):
        if self.iblank is None:
            m = jnp.all((self.data==0)|(self.data==1),axis=0)
        else:
            m = (jnp.arange(self.data.shape[1]) >= self.iblank)[:,None]
        z = (self.data-self.mean)/self.stdev     
        mask2d = jnp.broadcast_to(m,self.data.T.shape).T
        self.norm = jnp.where(mask2d,self.data,z) # keep bindary cols
        return self.norm

    def unnormalize(self):
        if self.iblank is None:
            m = jnp.all((self.norm==0)|(self.norm==1),axis=0)
        else:
            m = (jnp.arange(self.norm.shape[1]) >= self.iblank)[:,None]
        denorm = (self.norm * self.stdev) + self.mean 
        mask2d = jnp.broadcast_to(m,self.norm.T.shape).T
        self.data = jnp.where(mask2d,self.norm,denorm)
        return self.data

    def split(self,nsplit):
        dataS = dataset(self.data[:nsplit,:,:],iblank=self.iblank)
        testS = dataset(self.data[nsplit:,:,:])
        output = datasetSet(dataS)
        output.addDset(testS,"test")
        return output

    def select(self,nodes=[0]):
        copy = deepcopy(self)
        copy.norm = copy.norm[:,nodes,:]
        copy.data = copy.data[:,nodes,:]
        copy.labels = copy.labels[nodes]
        copy.limits = copy.limits[nodes,:]
        copy.mean = copy.mean[:,nodes,:]
        copy.stdev = copy.stdev[:,nodes,:]
        copy.nodes_max = len(nodes)
        copy.node_ids = jnp.arange(copy.nodes_max)
        return copy
    
    def generate(self,datakey,Npts,norm=True):
        if norm == True:
            return jax.random.choice(datakey, self.norm, shape=(Npts,), replace=False)
        else:
            return jax.random.choice(datakey, self.data, shape=(Npts,), replace=False)

    def generate_split(self,datakey,Npts,norm=True):
        if norm == True:
            vals = jax.random.choice(datakey, self.norm, shape=(Npts,), replace=False)
            return vals[:,:self.iblank,:], vals[:,self.iblank:,:]
        else:
            vals = jax.random.choice(datakey, self.data, shape=(Npts,), replace=False)
            return vals[:,:self.iblank,:], vals[:,self.iblank:,:]

    def const_mod(self,inds,dnr=[]):

        # (0) Remove the removal indecies from the dataset
        mask = (~jnp.isin(self.removed, jnp.array(dnr)))|(self.removed>=self.iblank)
        self.removed = self.removed[mask]
        self.limits = jnp.delete(self.limits, self.removed, axis=0)
        self.labels = np.delete(self.labels, self.removed, axis=0)
        self.norm = jnp.delete(self.norm, self.removed, axis=1)
        self.mean = jnp.delete(self.mean, self.removed, axis=1)
        self.stdev = jnp.delete(self.stdev, self.removed, axis=1)
        self.data = jnp.delete(self.data, self.removed, axis=1)
        self.nodes_max = self.data.shape[1]
        self.node_ids = jnp.arange(self.nodes_max)
        self.iblank = self.iblank - jnp.sum((self.iblank*jnp.ones_like(self.removed)>self.removed))

        # (1) sort the removals list in order
        removed_index = jnp.sort(self.removed)

        # (2) for each removed index
        for ii in range(len(removed_index)):
            index = removed_index[ii]

            for comp in inds:
                for part in comp:
                    if index in part:
                        # (a) remove it from all lists
                        part.remove(index)

                    # (b) take one from all items greater than it
                    part[:] = [n-1 if n>index else n for n in part]

            # (c) subtract one from the removed indecies list
            removed_index = removed_index - 1
            
        return inds

    def blank_mod(self,inds,delblank=True):
        total_comps = sum([len(comp_type) for comp_type in inds])
        ind_end = max([max([max(comp) for comp in comp_type]) for comp_type in inds])

        # Find which components the blanks correspond to
        blanks = jnp.ones((self.data.shape[0],total_comps,1))
        cc = 0
        for comp in inds:
            for part in comp:
                #print(self.labels[part])
                # (a) separate the dataset into arrays corresponding to indices
                temp = self.data[:,part,:]

                # (b) see if the component contains nan
                mask = jnp.any(jnp.isnan(temp), axis=1)
                contain_nan = jnp.where(mask)[0]
                blanks = blanks.at[contain_nan,cc,0].set(0)
                #print(contain_nan)

                cc += 1

        indlist = [[list(range(ind_end+1,ind_end+1+total_comps))]]

        # Set relevant components in dataset to zero
        self.data = jnp.nan_to_num(self.data, nan=0.0)
        self.norm = jnp.nan_to_num(self.norm, nan=0.0)
        # WARNING: this also sets anything that is nan in the dataset for some reason (e.g. stdev = 0 to 0)
        # However, the normalization has a thing that if stdev is zero it will just divide by 1 instead

        # Add in the blanking features
        self.data = jnp.concatenate([self.data,blanks],axis=1)
        self.norm = jnp.concatenate([self.norm,blanks],axis=1)
        self.mean = jnp.concatenate([self.mean,jnp.zeros((1,blanks.shape[1],1))],axis=1)
        self.stdev = jnp.concatenate([self.stdev,jnp.ones((1,blanks.shape[1],1))],axis=1)
        lim_blank = jnp.vstack([jnp.zeros((total_comps)),jnp.ones((total_comps))]).T
        self.limits = jnp.concatenate([self.limits,lim_blank],axis=0)
        self.labels = np.append(self.labels,np.array(indlist, dtype=str))

        # Modify the indices accordingly
        inds_new = inds + indlist

        # Remove constant parts
        zero_mask = jnp.all(self.data == 0, axis=0)
        one_mask  = jnp.all(self.data == 1, axis=0)
        zero_cols = jnp.where(zero_mask)[0]
        one_cols  = jnp.where(one_mask)[0]
        if delblank == True:
            removal = jnp.hstack([zero_cols,one_cols])
        else:
            removal = one_cols
        
        self.removed = jnp.unique(jnp.hstack([self.removed,removal]))

        return inds_new

class datasetSet:
    def __init__(self,main):
        self.data = main.data
        self.mean = main.mean
        self.stdev = main.stdev
        self.main = main
        self.sets = {}
    
    def addDset(self,dsetin,tag,norm=False):
        dsetin.mean = self.mean
        dsetin.stdev = self.stdev
        dsetin.iblank = self.main.iblank
        if norm == False:
            dsetin.normalize()
        if norm == True:
            dsetin.unnormalize()
        self.sets[tag] = dsetin

    def select(self,nodes=[0]):
        copy = deepcopy(self)
        copy.main = copy.main.select(nodes)
        copy.data = copy.main.data
        copy.mean = copy.mean
        copy.stdev = copy.stdev
        for key, value in copy.sets.items():
            copy.sets[key] = value.select(nodes)

        return copy
