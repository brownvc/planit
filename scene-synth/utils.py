import gzip
import math
import os
import os.path
import sys
import pickle
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from contextlib import contextmanager
import sys

'''
Find the first item in a list that satisfies a predicate
'''
def find(pred, lst):
    for item in lst:
        if pred(item):
            return item

'''
Ensure a directory exists
'''
def ensuredir(dirname):
    """
    Ensure a directory exists
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def zeropad(num, n):
    """
    Turn a number into a string that is zero-padded up to length n
    """
    sn = str(num)
    while len(sn) < n:
        sn = '0' + sn
    return sn

def pickle_dump_compressed(object, filename, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Pickles + compresses an object to file
    """
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, protocol))
    file.close()

def pickle_load_compressed(filename):
    """
    Loads a compressed pickle file and returns reconstituted object
    """
    file = gzip.GzipFile(filename, 'rb')
    buffer = b""
    while True:
        data = file.read()
        if data == b"":
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object

def get_data_root_dir():
    """
    Gets the root dir of the dataset
    Check env variable first,
    if not set, use the {code_location}/data
    """
    env_path = os.environ.get("SCENESYNTH_DATA_PATH")
    if env_path:
    #if False: #Debug purposes
        return env_path
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        return f"{root_dir}/data"

@contextmanager
def stdout_redirected(to=os.devnull):
    """
    From https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    Suppress C warnings
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
# Turn index into one-hot vector
# indices = torch.LongTensor of indices (batched)
def index_to_onehot(indices, numclasses):
    b_size = indices.size()[0]
    one_hot = torch.zeros(b_size, numclasses)
    for i in range(0, b_size):
        cat_index = indices[i][0]
        one_hot[i][cat_index] = 1
    return one_hot

# A better version of index to one-hot that uses scatter
def index_to_onehot_fast(indices, numclasses):
	indices = indices.unsqueeze(-1)
	onehot = torch.zeros(indices.shape[0], numclasses)
	if indices.is_cuda:
		onehot = onehot.cuda()
	onehot.scatter_(1, indices, 1)
	return onehot

# Do nearest-neighbor downsampling on an image tensor with
#    arbitrary channels
# Img must be a 4D B x C x H x W tensor
# Factor must be an integer (is converted to one)
def nearest_downsample(img, factor):
	assert (factor > 0 and factor == int(factor)), 'Downsample factor must be positive integer'
	# We do this by convolving with a strided convolutional kernel, whose kernel
	#    is size one and is the identity matrix between input and output channels
	nchannels = img.shape[1]
	kernel = torch.eye(nchannels).view(nchannels, nchannels, 1, 1)
	if img.is_cuda:
		kernel = kernel.cuda()
	return F.conv2d(img, weight=kernel, stride=factor)

# Softmax across the pixels of an image batch
def softmax2d(img):
    size = img.shape[2]
    img = img.view(img.shape[0], -1)
    img = F.softmax(img, dim=-1)
    img = img.view(-1, size, size)
    return img

# Decorator to memoize a function
# https://medium.com/@nkhaja/memoization-and-decorators-with-python-32f607439f84
def memoize(func):
	cache = func.cache = {}

	@functools.wraps(func)
	def memoized_func(*args, **kwargs):
		key = str(args) + str(kwargs)
		if key not in cache:
			cache[key] = func(*args, **kwargs)
		return cache[key]

	return memoized_func

if __name__ == "__main__":
    print(get_data_root_dir())


# Converting image batch tensors that represent binary masks into outlines
#    and signed distance functions
edge_filters_x = torch.tensor([
    [0, 0, 0],
    [-1, 0, 1],
    [0, 0, 0],
]).float()
edge_filters_y = torch.tensor([
    [0, -1, 0],
    [0, 0, 0],
    [0, 1, 0],
]).float()
def mask_to_outline(img):
	num_channels = img.shape[1]
	efx = torch.zeros(num_channels, num_channels, 3, 3)
	efy = torch.zeros(num_channels, num_channels, 3, 3)
	for i in range(num_channels):
		efx[i, i, :, :] = edge_filters_x
		efy[i, i, :, :] = edge_filters_y
	if img.is_cuda:
		efx = efx.cuda()
		efy = efy.cuda()
	edges_x = F.conv2d(img, efx, padding=1)
	edges_y = F.conv2d(img, efy, padding=1)
	edges = edges_x + edges_y
	return (edges != 0).float()

from scipy.ndimage import distance_transform_edt
def mask_to_sdf(img):
	outline = mask_to_outline(img)
	outline_neg = 1 - outline
	dists = torch.tensor(distance_transform_edt(outline_neg)).float()
	diag_len = math.sqrt(2*img.shape[2]*img.shape[2])
	dists = dists / diag_len    # normalize
	return torch.where(img > 0, -dists, dists)

# A wrapper class that forwards all method calls on to one or more 'forwardee' objects
class Forwarder:
    def __init__(self, *args):
        self.forwardees = list(args)
    def __getattr__(self, name):
        def method(*args):
            ret = None
            for fwdee in self.forwardees:
                f = getattr(fwdee, name)
                assert callable(f), f'{name} is not a method; cannot forward'
                fret = f(*args)
                assert (ret is None or fret == ret), 'Forwarded method calls had different return values'
                ret = fret
        return method
