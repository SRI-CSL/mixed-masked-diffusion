import jax
import jax.numpy as jnp

class dataloader:
    """Lightweight JAX dataset wrapper that generates per-device
    training and test batches. Train-test splitting and pmap
    parallelism.
    """

    def __init__(self,train_args,datagroup):
        self.train = datagroup.main
        self.test = datagroup.sets["test"]
        self.batch_size = train_args["batch_size"]
        self.n_devices = train_args["n_devices"]
        self.key = train_args["rng"]
    
    def load_train_split(self):
        key, rng_data = jax.random.split(self.key)
        xs, mask = self.train.generate_split(rng_data,self.batch_size*self.n_devices)
        xs_reshape = xs.reshape(self.n_devices, self.batch_size, *xs.shape[1:])
        mask_reshape = mask.reshape(self.n_devices, self.batch_size, *mask.shape[1:])
        batch = {
            "x": xs_reshape,
            "mask": mask_reshape
        }
        self.key = key
        return batch

    def load_test_split(self):
        key, rng_data = jax.random.split(self.key)
        xs, mask = self.test.generate_split(rng_data,self.batch_size*self.n_devices)
        xs_reshape = xs.reshape(self.n_devices, self.batch_size, *xs.shape[1:])
        mask_reshape = mask.reshape(self.n_devices, self.batch_size, *mask.shape[1:])
        batch = {
            "x": xs_reshape,
            "mask": mask_reshape
        }
        self.key = key
        return batch

    def load_train(self):
        key, rng_data = jax.random.split(self.key)
        xs = self.train.generate(rng_data,self.batch_size*self.n_devices)
        xs_reshape = xs.reshape(self.n_devices, self.batch_size, *xs.shape[1:])
        batch = {
            "x": xs_reshape
        }
        self.key = key
        return batch
    
    def load_test(self):
        key, rng_data = jax.random.split(self.key)
        xs = self.test.generate(rng_data,self.batch_size*self.n_devices)
        xs_reshape = xs.reshape(self.n_devices, self.batch_size, *xs.shape[1:])
        batch = {
            "x": xs_reshape
        }
        self.key = key
        return batch