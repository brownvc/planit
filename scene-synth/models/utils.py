import math
import torch
import torch.nn as nn
import torch.nn.functional as F


## ------------------------------------------------------------------------------------------------


# Make reshape into a layer, so we can put it in nn.Sequential
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

# Render a signed distance field image for a given oriented box
# img_size: a 2-tuple of image dimensions
# box_dims: a 2D tensor of x,y box lengths
# loc: a 2D tensor location
# orient: a 2D tensor (sin, cos) orientation
# NOTE: This operates on batches
# NOTE: Since box_dims are expressed in percentage of image size, everything is set
#    up to work in that coordinate system. In particular, this means that locations are
#    rescaled from [-1, 1] to [0, 1]
def render_obb_sdf(img_size, box_dims, loc, orient):
    batch_size = loc.shape[0]

    # I don't know why, but the orient dims need to be reversed...
    orient = torch.stack([orient[:, 1], orient[:, 0]], dim=-1)

    # Rescale location
    loc = (loc + 1) * 0.5

    # Rescale the box dims to be half-widths
    r = box_dims * 0.5

    # Create coordinate grid
    linear_points_x = torch.linspace(0, 1, img_size[0])
    linear_points_y = torch.linspace(0, 1, img_size[1])
    coords = torch.Tensor(batch_size, img_size[0], img_size[1], 3)
    coords[:, :, :, 0] = torch.ger(linear_points_x, torch.ones(img_size[0]))
    coords[:, :, :, 1] = torch.ger(torch.ones(img_size[1]), linear_points_y)
    coords[:, :, :, 2] = 1     # Homogeneous coords
    # (Non-standard ordering of dimensions (x, y, b, 3) in order to support broadcasting)
    coords = coords.permute(1, 2, 0, 3)
    if loc.is_cuda:
        coords = coords.cuda()

    # Attempting to do this faster by inverse-transforming the coordinate points and
    #    then evaluating the SDF for a standard axis-aligned box
    inv_rot_matrices = torch.zeros(batch_size, 3, 3)
    if loc.is_cuda:
        inv_rot_matrices = inv_rot_matrices.cuda()
    cos = orient[:, 0]
    sin = orient[:, 1]
    inv_rot_matrices[:, 0, 0] = cos
    inv_rot_matrices[:, 1, 1] = cos
    inv_rot_matrices[:, 0, 1] = sin
    inv_rot_matrices[:, 1, 0] = -sin
    inv_rot_matrices[:, 2, 2] = 1
    inv_trans_matrices = [torch.eye(3, 3) for i in range(0, batch_size)]
    inv_trans_matrices = torch.stack(inv_trans_matrices)
    if loc.is_cuda:
        inv_trans_matrices = inv_trans_matrices.cuda()
    inv_trans_matrices[:, 0, 2] = -loc[:, 0]
    inv_trans_matrices[:, 1, 2] = -loc[:, 1]
    inv_matrices = torch.matmul(inv_rot_matrices, inv_trans_matrices)
    # Discard the last row (to get 2x3 matrices)
    inv_matrices = inv_matrices[:, 0:2, :]
    # Multiply the coords by these matrices
    # Batch matrix multiply of (b, 2, 3) by (x, y, b, 3, 1) -> (x, y, b, 2, 1)
    coords = coords.view(img_size[0], img_size[1], batch_size, 3, 1)
    coords = torch.matmul(inv_matrices, coords)
    coords = coords.view(img_size[0], img_size[1], batch_size, 2)

    # Now evaluate the SDF of an axis-aligned box centered at the origin
    xdist = coords[:, :, :, 0].abs() - r[:, 0]
    ydist = coords[:, :, :, 1].abs() - r[:, 1]
    dist = torch.max(xdist, ydist)
    dist = dist.permute(2, 0, 1)
    return dist.view(batch_size, 1, img_size[0], img_size[1])

## ------------------------------------------------------------------------------------------------

###### Modules for Feature-wise Linear Modulation ######
# https://arxiv.org/pdf/1709.07871.pdf

class FiLM(nn.Module):

	def __init__(self, n_features):
		super(FiLM, self).__init__()
		self.n_features = n_features

	def forward(self, x, gammas, betas):
		# If x is a batch of flat tensors, easy
		# If x is a batch of image tensors, then we need to expand the params
		#    so that pointwise mult and add work
		if len(x.shape) == 4:
			#print(gammas.shape) # batch_size * 1 * n_features
			gammas = gammas.view(-1, self.n_features, 1, 1).expand_as(x)
			betas = betas.view(-1, self.n_features, 1, 1).expand_as(x)
		return (gammas * x) + betas

# A version of nn.Sequential that can control what happens at each step
class CustomSequential(nn.Sequential):

	def forward(self, x, *args):
		self.init(x, *args)
		i = 0
		for module in self._modules.values():
			x = self.step(x, module, i, *args)
			i = i + 1
		return x

	# Called at the beginning of forward (does nothing by default)
	def init(self, x, *args):
		pass

	# Called on each step (just forwards the input through the module by default)
	def step(self, x, module, i, *args):
		return module(x)

# Like nn.Sequential, but takes an extra argument that gets passed through
#    a linear layer to create FiLM parameters. Number of parameters to create
#    is determined by looking at the FiLM modules in the sequence.
class FiLMSequential(CustomSequential):

	def __init__(self, cond_inp_size, *args):
		super(FiLMSequential, self).__init__(*args)

		# Figure out how many FiLM modules there are and what their sizes are
		film_module_sizes = list(map(lambda m: m.n_features, \
			filter(lambda m: isinstance(m, FiLM), list(self._modules.values()))))
		self.film_module_nparams = list(map(lambda x: 2 * x, film_module_sizes))
		film_total_nparams = sum(self.film_module_nparams)

		self.n_modules = len(list(self._modules.values()))

		# Linear layer to convert conditioning input into FiLM params
		self.linear = nn.Linear(cond_inp_size, film_total_nparams)

	def init(self, x, cond_inp):
		# Convert the cond_inp into FiLM params
		film_params = self.linear(cond_inp)
		# Split these into blocks, one per FiLM module
		self.film_module_params = torch.split(film_params, self.film_module_nparams, dim=-1)
		# Initialize a counter to keep track of which FiLM module we're on
		self.f_idx = 0

	def step(self, x, module, i, cond_inp):
		# Stop if we've evaluated all the sequential modules already
		if i == self.n_modules:
			return x
		if isinstance(module, FiLM):
			# Split out the param block into betas and gammas
			params = self.film_module_params[self.f_idx]
			gammas, betas = torch.split(params, params.shape[-1]//2, dim=-1)
			x = module(x, gammas, betas)
			self.f_idx += 1
		else:
			x = module(x)
		return x

# Takes an input and a conditioning input
# Runs the conditioning input through some network, then passes it
#    along with the input into a second network, which is assumed to be
#    a FiLMSequential
class FiLMNetworkPair(nn.Module):

	def __init__(self, cond_net, filmed_net):
		super(FiLMNetworkPair, self).__init__()
		self.cond_net = cond_net
		self.filmed_net = filmed_net

	def forward(self, inp, cond_inp):
		cond_inp = self.cond_net(cond_inp)
		return self.filmed_net(inp, cond_inp)

def log(x):
    return torch.log(torch.max(x, 1e-5))

def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad

def make_walls_img(dims):
    wall_img = torch.zeros(1, img_size, img_size)
    w = int((dims[0] / 2) * img_size)
    h = int((dims[1] / 2) * img_size)
    mid = img_size // 2
    wall_img[0, (mid+h):(mid+h+wall_thickness), 0:(mid+w+wall_thickness)] = 1 # Horiz wall
    wall_img[0, 0:(mid+h+wall_thickness), (mid+w):(mid+w+wall_thickness)] = 1 # Vert wall
    return wall_img
def make_walls_img_batch(dims_batch):
    batch_size = dims_batch.shape[0]
    return torch.stack([make_walls_img(dims_batch[i]) for i in range(batch_size)], dim=0)

def unitnormal_normal_kld(mu, logvar, size_average=True):
    # Always reduce along the data dimensionality axis
    # output = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    output = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    # Average along the batch axis, if requested
    if size_average:
        output = torch.mean(output)
    return output

def inverse_xform_img(img, loc, orient, output_size):
    batch_size = img.shape[0]
    matrices = torch.zeros(batch_size, 2, 3).cuda()
    cos = orient[:, 0]
    sin = orient[:, 1]
    matrices[:, 0, 0] = cos
    matrices[:, 1, 1] = cos
    matrices[:, 0, 1] = -sin
    matrices[:, 1, 0] = sin
    matrices[:, 0, 2] = loc[:, 1]
    matrices[:, 1, 2] = loc[:, 0]
    out_size = torch.Size((batch_size, img.shape[1], output_size, output_size))
    grid = F.affine_grid(matrices, out_size)
    return F.grid_sample(img, grid)

def forward_xform_img(img, loc, orient, output_size):
    # First, build the inverse rotation matrices
    batch_size = img.shape[0]
    inv_rot_matrices = torch.zeros(batch_size, 3, 3).cuda()
    cos = orient[:, 0]
    sin = orient[:, 1]
    inv_rot_matrices[:, 0, 0] = cos
    inv_rot_matrices[:, 1, 1] = cos
    inv_rot_matrices[:, 0, 1] = sin
    inv_rot_matrices[:, 1, 0] = -sin
    inv_rot_matrices[:, 2, 2] = 1
    # Then, build the inverse translation matrices
    # (Apparently, x and y need to be swapped. I don't know why...)
    inv_trans_matrices = [torch.eye(3, 3) for i in range(0, batch_size)]
    inv_trans_matrices = torch.stack(inv_trans_matrices).cuda()
    inv_trans_matrices[:, 0, 2] = -loc[:, 1]
    inv_trans_matrices[:, 1, 2] = -loc[:, 0]
    # Multiply them to get the full affine matrix
    inv_matrices = torch.matmul(inv_rot_matrices, inv_trans_matrices)
    # Discard the last row (affine_grid expects 2x3 matrices)
    inv_matrices = inv_matrices[:, 0:2, :]
    # Finalize
    out_size = torch.Size((batch_size, img.shape[1], output_size, output_size))
    grid = F.affine_grid(inv_matrices, out_size)
    return F.grid_sample(img, grid)

def default_loc_orient(batch_size):
    loc = torch.zeros(batch_size, 2).cuda()
    orient = torch.stack([torch.Tensor([math.cos(0), math.sin(0)]) for i in range(batch_size)], dim=0).cuda()
    return loc, orient

class DownConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(DownConvBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, stride=2, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(outplanes)
        self.activation = nn.ReLU()
    def forward(self, x):
        # return self.activation(self.conv(x))
        return self.activation(self.bn(self.conv(x)))

def make_walls_img(dims):
    wall_img = torch.zeros(1, img_size, img_size)
    w = int((dims[0] / 2) * img_size)
    h = int((dims[1] / 2) * img_size)
    mid = img_size // 2
    wall_img[0, (mid+h):(mid+h+wall_thickness), 0:(mid+w+wall_thickness)] = 1 # Horiz wall
    wall_img[0, 0:(mid+h+wall_thickness), (mid+w):(mid+w+wall_thickness)] = 1 # Vert wall
    # wall_img[0, :, (mid+w):(mid+w+wall_thickness)] = 1 # Vert wall
    return wall_img

def render_orientation_sdf(img_size, dims, loc, orient):
    batch_size = loc.shape[0]

    # I don't know why, but the orient dims need to be reversed...
    orient = torch.stack([orient[:, 1], orient[:, 0]], dim=-1)

    # Rescale location
    loc = (loc + 1) * 0.5

    # Rescale the box dims to be half-widths
    r = dims * 0.5

    # Create coordinate grid
    linear_points_x = torch.linspace(0, 1, img_size[0])
    linear_points_y = torch.linspace(0, 1, img_size[1])
    coords = torch.Tensor(batch_size, img_size[0], img_size[1], 3)
    coords[:, :, :, 0] = torch.ger(linear_points_x, torch.ones(img_size[0]))
    coords[:, :, :, 1] = torch.ger(torch.ones(img_size[1]), linear_points_y)
    coords[:, :, :, 2] = 1     # Homogeneous coords
    # (Non-standard ordering of dimensions (x, y, b, 3) in order to support broadcasting)
    coords = coords.permute(1, 2, 0, 3)
    if loc.is_cuda:
        coords = coords.cuda()

    # Attempting to do this faster by inverse-transforming the coordinate points and
    #    then evaluating the SDF for a standard axis-aligned plane
    inv_rot_matrices = torch.zeros(batch_size, 3, 3)
    if loc.is_cuda:
        inv_rot_matrices = inv_rot_matrices.cuda()
    cos = orient[:, 0]
    sin = orient[:, 1]
    inv_rot_matrices[:, 0, 0] = cos
    inv_rot_matrices[:, 1, 1] = cos
    inv_rot_matrices[:, 0, 1] = sin
    inv_rot_matrices[:, 1, 0] = -sin
    inv_rot_matrices[:, 2, 2] = 1
    inv_trans_matrices = [torch.eye(3, 3) for i in range(0, batch_size)]
    inv_trans_matrices = torch.stack(inv_trans_matrices)
    if loc.is_cuda:
        inv_trans_matrices = inv_trans_matrices.cuda()
    inv_trans_matrices[:, 0, 2] = -loc[:, 0]
    inv_trans_matrices[:, 1, 2] = -loc[:, 1]
    inv_matrices = torch.matmul(inv_rot_matrices, inv_trans_matrices)
    # Discard the last row (to get 2x3 matrices)
    inv_matrices = inv_matrices[:, 0:2, :]
    # Multiply the coords by these matrices
    # Batch matrix multiply of (b, 2, 3) by (x, y, b, 3, 1) -> (x, y, b, 2, 1)
    coords = coords.view(img_size[0], img_size[1], batch_size, 3, 1)
    coords = torch.matmul(inv_matrices, coords)
    coords = coords.view(img_size[0], img_size[1], batch_size, 2)

    # Now evaluate the SDF of an axis-aligned plane
    # (Signed dist to plane is just the x coordinate)
    dist = coords[:, :, :, 0] 
    dist = dist.permute(2, 0, 1)
    return dist.view(batch_size, 1, img_size[0], img_size[1])

    # xdist = (coords[:, :, :, 0].abs() - r[:, 0]).max(torch.Tensor([0]).cuda()) * coords[:, :, :, 0].sign()
    # ydist = (coords[:, :, :, 1].abs() - r[:, 1]).max(torch.Tensor([0]).cuda()) * coords[:, :, :, 1].sign()
    # dist = torch.stack([xdist, ydist], dim=-1)
    # dist = dist.permute(2, 3, 0, 1)
    # return dist

# Like stack the regular bbox SDF with the orientation plane SDF
def render_oriented_sdf(img_sizes, dims, loc, orient):
    sdf = render_obb_sdf(img_sizes, dims, loc, orient)
    osdf = render_orientation_sdf(img_sizes, dims, loc, orient)
    # sdf = render_obb_sdf_slow(img_sizes, dims, loc, orient)
    # osdf = render_orientation_sdf_slow(img_sizes, dims, loc, orient)
    return torch.cat([sdf, osdf], dim=1)

CARDINAL_ANGLES = torch.Tensor([0, math.pi/2, math.pi, 3*math.pi/2])
CARDINAL_DIRECTIONS = torch.stack([CARDINAL_ANGLES.cos(), CARDINAL_ANGLES.sin()], dim=1)

# Snap an orientation to its nearest cardinal direction
def snap_orient(orient):
    sims = [F.cosine_similarity(orient, cdir.unsqueeze(0).cuda()) for cdir in CARDINAL_DIRECTIONS]
    sims = torch.stack(sims, dim=1)
    maxvals, indices = sims.max(dim=1)
    return CARDINAL_DIRECTIONS[indices].cuda()

def should_snap(orient):
    snap_sims = [F.cosine_similarity(orient, cdir.unsqueeze(0)) for cdir in CARDINAL_DIRECTIONS]
    snap_sims = torch.stack(snap_sims, dim=1)
    snap_sim, _ = snap_sims.max(dim=1)
    snap = (snap_sim > (1 - 1e-4)).float()
    return snap

def index_to_onehot(indices, numclasses):
    """
    Turn index into one-hot vector
    indices = torch.LongTensor of indices (batched)
    """
    b_size = indices.size()[0]
    one_hot = torch.zeros(b_size, numclasses)
    for i in range(0, b_size):
        cat_index = indices[i][0]
        one_hot[i][cat_index] = 1
    return one_hot

def index_to_onehot_fast(indices, numclasses):
    """A better version of index to one-hot that uses scatter"""
    indices = indices.unsqueeze(-1)
    onehot = torch.zeros(indices.shape[0], numclasses)
    if indices.is_cuda:
        onehot = onehot.cuda()
    onehot.scatter_(1, indices, 1)
    return onehot

def nearest_downsample(img, factor):
    """
    Do nearest-neighbor downsampling on an image tensor with
       arbitrary channels
    Img must be a 4D B x C x H x W tensor
    Factor must be an integer (is converted to one)
    """
    assert (factor > 0 and factor == int(factor)), 'Downsample factor must be positive integer'
    # We do this by convolving with a strided convolutional kernel, whose kernel
    #    is size one and is the identity matrix between input and output channels
    nchannels = img.shape[1]
    kernel = torch.eye(nchannels).view(nchannels, nchannels, 1, 1)
    if img.is_cuda:
            kernel = kernel.cuda()
    return F.conv2d(img, weight=kernel, stride=factor)

def softmax2d(img):
    """Softmax across the pixels of an image batch"""
    size = img.shape[2]
    img = img.view(img.shape[0], -1)
    img = F.softmax(img, dim=-1)
    img = img.view(-1, size, size)
    return img

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

def mask_to_sdf(img):
    outline = mask_to_outline(img)
    outline_neg = 1 - outline
    dists = torch.tensor(distance_transform_edt(outline_neg)).float()
    diag_len = math.sqrt(2*img.shape[2]*img.shape[2])
    dists = dists / diag_len    # normalize
    return torch.where(img > 0, -dists, dists)

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