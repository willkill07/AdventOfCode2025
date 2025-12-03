import nvtx

rng = range

def arng(self, *args, **kwargs):
    return self.arange(*args, **kwargs)

def cast_to(self, *args, **kwargs):
    return self.astype(*args, **kwargs)

def cond(self, *args, **kwargs):
    return self.where(*args, **kwargs)

def do_pow(self, *args, **kwargs):
    return self.power(*args, **kwargs)

def init_array_z(self, *args, **kwargs):
    return self.zeros_like(*args, **kwargs)

def mkarray(self, shape, dtype):
    return self.empty(shape, dtype=dtype)

def nonz(self, *args, **kwargs):
    return self.nonzero(*args, **kwargs)

prof = nvtx.annotate
