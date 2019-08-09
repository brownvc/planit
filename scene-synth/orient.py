import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
import torchvision
import numpy as np
import random
import math
from models.utils import *
from latent_dataset import LatentDataset
import utils
import os
from data import ObjectCategories

# ---------------------------------------------------------------------------------------
img_size = 64
latent_size = 10
hidden_size = 40
output_size = 2
# ---------------------------------------------------------------------------------------

class Model(nn.Module):

    def make_net_fn(self, netdict, makefn):
        def net_fn(cat):
            # We assume that the data loader has extracted the single category index
            #    that we'll be using for this batch
            cat = str(cat)
            if cat in netdict:
                return netdict[cat]
            else:
                net = makefn().cuda()
                netdict[cat] = net
                return net
        return net_fn
    
    def __init__(self, latent_size, hidden_size, num_input_channels):
        super(Model, self).__init__()
        self.snapping = False
        self.latent_size = latent_size
        self.testing = False

        def make_encoder():
            return nn.Sequential(
                nn.Linear(2, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, 2*latent_size)
            )

        def make_cond_prior():
            return nn.Sequential(
                # 64 -> 32
                DownConvBlock(num_input_channels, 8),
                # 32 -> 16
                DownConvBlock(8, 16),
                # 16 -> 8
                DownConvBlock(16, 32),
                # 8 -> 4
                DownConvBlock(32, 64),
                # 4 -> 1
                nn.AdaptiveAvgPool2d(1),
                # Final linear layer
                Reshape(-1, 64),
                nn.Linear(64, latent_size)
            )
        
        def make_snap_predictor():
            return nn.Sequential(
                # 64 -> 32
                DownConvBlock(num_input_channels, 8),
                # 32 -> 16
                DownConvBlock(8, 16),
                # 16 -> 8
                DownConvBlock(16, 32),
                # 8 -> 4
                DownConvBlock(32, 64),
                # 4 -> 1
                nn.AdaptiveAvgPool2d(1),
                # Final linear layer
                Reshape(-1, 64),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def make_generator():
            return nn.Sequential(
                nn.Linear(2*latent_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, 2)
            )

        def make_discriminator():
            return nn.Sequential(
                # 64 -> 32
                DownConvBlock(num_input_channels+2, 8),
                # 32 -> 16
                DownConvBlock(8, 16),
                # 16 -> 8
                DownConvBlock(16, 32),
                # 8 -> 4
                DownConvBlock(32, 64),
                # 4 -> 1
                nn.AdaptiveAvgPool2d(1),
                # Final linear layer
                Reshape(-1, 64),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        self.encoders = nn.ModuleDict()
        self.cond_priors = nn.ModuleDict()
        self.snap_predictors = nn.ModuleDict()
        self.generators = nn.ModuleDict()
        self.discriminators = nn.ModuleDict()

        self.encoder = self.make_net_fn(self.encoders, make_encoder)
        self.cond_prior = self.make_net_fn(self.cond_priors, make_cond_prior)
        self.snap_predictor = self.make_net_fn(self.snap_predictors, make_snap_predictor)
        self.generator = self.make_net_fn(self.generators, make_generator)
        self.discriminator = self.make_net_fn(self.discriminators, make_discriminator)

    def encode(self, t_orient, cat):
        mu_logvar = self.encoder(cat)(t_orient)
        return torch.split(mu_logvar, self.latent_size, dim=1)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        gdis = torch.distributions.Normal(mu, std)
        return gdis.rsample()

    def generate(self, noise, walls, cat):
        enc_walls = self.cond_prior(cat)(walls)
        gen_out = self.generator(cat)(torch.cat([noise, enc_walls], dim=1))
        go1, go2 = torch.split(gen_out, 1, dim=1)
        orient_x = F.tanh(go1)
        orient_y = torch.sqrt(1.0 - orient_x * orient_x)
        y_sign_p = F.sigmoid(go2)
        if self.testing:
            y_sign = torch.where(y_sign_p > 0.5, torch.ones_like(orient_y), -torch.ones_like(orient_y))
            orient_y *= y_sign
            orient = torch.stack([orient_x, orient_y], dim=1).squeeze()
            if len(list(orient.size())) == 1:
                orient = orient.unsqueeze(0)
            if self.snapping:
                do_snap_p = self.snap_predict(walls, cat)
                do_snap = do_snap_p > 0.5
                orient = torch.where(do_snap, snap_orient(orient), orient)
            return orient
        else:
            return orient_x, y_sign_p
        

    def snap_predict(self, walls, cat):
        return self.snap_predictor(cat)(walls)

    def discriminate(self, walls, loc, orient, dims, cat):
        sdf = render_oriented_sdf((img_size, img_size), dims, loc, orient)
        return self.discriminator(cat)(torch.cat([sdf, walls], dim=1))

    def set_requires_grad(self, phase, cat):
        if phase == 'D':
            set_requires_grad(self.generator(cat), False)
            set_requires_grad(self.discriminator(cat), True)
            set_requires_grad(self.encoder(cat), False)
            set_requires_grad(self.cond_prior(cat), False)
            set_requires_grad(self.snap_predictor(cat), False)
        elif phase == 'G':
            set_requires_grad(self.generator(cat), True)
            set_requires_grad(self.discriminator(cat), False)
            set_requires_grad(self.encoder(cat), False)
            set_requires_grad(self.cond_prior(cat), True)
            set_requires_grad(self.snap_predictor(cat), False)
        elif phase == 'VAE':
            set_requires_grad(self.generator(cat), True)
            set_requires_grad(self.discriminator(cat), False)
            set_requires_grad(self.encoder(cat), True)
            set_requires_grad(self.cond_prior(cat), True)
            set_requires_grad(self.snap_predictor(cat), False)
        elif phase == 'snap':
            set_requires_grad(self.generator(cat), False)
            set_requires_grad(self.discriminator(cat), False)
            set_requires_grad(self.encoder(cat), False)
            set_requires_grad(self.cond_prior(cat), False)
            set_requires_grad(self.snap_predictor(cat), True)
        else:
            raise ValueError(f'Unrecognized phase {phase}')

    def save(self, filename):
        torch.save({
            'cats_seen': list(self.generators.keys()),
            'state': self.state_dict()
        }, filename)

    def load(self, filename):
        blob = torch.load(filename)
        for cat in blob['cats_seen']:
            _ = self.encoder(cat)
            _ = self.cond_prior(cat)
            _ = self.generator(cat)
            _ = self.discriminator(cat)
            _ = self.snap_predictor(cat)
        self.load_state_dict(blob['state'])

class Optimizers:

    def make_optimizer_fn(self, optimizers, list_of_netfns):
        this = self
        def optimizer_fn(cat):
            cat = str(cat)
            if cat in optimizers:
                return optimizers[cat]
            else:
                params = []
                for netfn in list_of_netfns:
                    params.extend(list(netfn(cat).parameters()))
                optimizer = optim.Adam(params, lr=this.lr)
                optimizers[cat] = optimizer
                return optimizer
        return optimizer_fn

    def __init__(self, model, lr):
        self.lr = lr
        self.g_optimizers = {}
        self.d_optimizers = {}
        self.e_optimizers = {}
        self.s_optimizers = {}
        self.g_optimizer = self.make_optimizer_fn(self.g_optimizers, [model.generator, model.cond_prior])
        self.d_optimizer = self.make_optimizer_fn(self.d_optimizers, [model.discriminator])
        self.e_optimizer = self.make_optimizer_fn(self.e_optimizers, [model.encoder])
        self.s_optimizer = self.make_optimizer_fn(self.s_optimizers, [model.snap_predictor])

    def save(self, filename):
        g_state = {cat : opt.state_dict() for cat, opt in self.g_optimizers.items()}
        d_state = {cat : opt.state_dict() for cat, opt in self.d_optimizers.items()}
        e_state = {cat : opt.state_dict() for cat, opt in self.e_optimizers.items()}
        s_state = {cat : opt.state_dict() for cat, opt in self.s_optimizers.items()}
        torch.save([g_state, d_state, e_state, s_state], filename)

    def load(self, filename):
        states = torch.load(filename)
        g_state = states[0]
        d_state = states[1]
        e_state = states[2]
        s_state = states[3]
        for cat,state in g_state:
            self.g_optimizer(cat).load_state_dict(state)
        for cat,state in d_state:
            self.d_optimizer(cat).load_state_dict(state)
        for cat,state in e_state:
            self.e_optimizer(cat).load_state_dict(state)
        for cat,state in s_state:
            self.s_optimizer(cat).load_state_dict(state)

# ---------------------------------------------------------------------------------------

if __name__ == '__main__':
    img_size = 64
    latent_size = 10
    hidden_size = 40
    output_size = 2
    batch_size = 8
    epoch_size = 5000
    valid_set_size = 160
    log_every = 50
    save_every = 5

    # num_epochs = 50
    # num_epochs = 0
    num_epochs = 1000

    use_jitter = False
    jitter_stdev = 0.01

    which_to_load = 500

    parser = argparse.ArgumentParser(description='orient')
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--data-folder', type=str, default="")
    args = parser.parse_args()
    outdir = f'./output/{args.save_dir}'
    utils.ensuredir(outdir)

    data_folder = args.data_folder

    data_root_dir = utils.get_data_root_dir()
    categories = ObjectCategories().all_non_arch_categories(data_root_dir, data_folder)
    num_categories = len(categories)
    num_input_channels = num_categories+8

    nc = num_categories

    # Dataset size is based on the number of available scenes - the valid set size
    dataset_size = len([f for f in os.listdir(f'{data_root_dir}/{data_folder}') \
        if f.endswith('.jpg')]) - valid_set_size
    dataset_size = int(dataset_size / batch_size) * batch_size

    logfile = open(f"{outdir}/log.txt", 'w')
    def LOG(msg):
        print(msg)
        logfile.write(msg + '\n')
        logfile.flush()

    dataset = LatentDataset(
        data_folder = data_folder,
        scene_indices = (0, dataset_size),
        use_same_category_batches = True,
        epoch_size=epoch_size
    )
    # Put this here right away in case the creation of the data loader reads and
    #    caches the length of the dataset
    dataset.prepare_same_category_batches(batch_size)
    # NOTE: *MUST* use shuffle = False here to guarantee that each batch has the
    #    only one category in it
    data_loader = data.DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = 6,
        shuffle = False
    )

    valid_dataset = LatentDataset(
        data_folder = data_folder,
        scene_indices = (dataset_size, dataset_size+valid_set_size),
        use_same_category_batches = True,
        epoch_size=valid_set_size
        # seed = 42
    )
    dataset.prepare_same_category_batches(batch_size)
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size = batch_size,
        num_workers = 6,
        shuffle = False
    )


    test_dataset = valid_dataset
    test_loader = data.DataLoader(
        valid_dataset,
        batch_size = batch_size,
        num_workers = 1,
        shuffle = False
    )


    model = Model(latent_size, hidden_size, num_input_channels).cuda()
    model.train()
    optimizers = Optimizers(model=model, lr=0.0005)

    def train(e):
        dataset.prepare_same_category_batches(batch_size)
        LOG(f'========================= EPOCH {e} =========================')
        for i, (input_img, output_mask, t_cat, t_loc, t_orient, t_dims, catcount) in enumerate(data_loader):
            t_cat = torch.squeeze(t_cat)
            # Verify that we've got only one category in this batch
            t_cat_0 = t_cat[0]
            assert((t_cat == t_cat_0).all())
            t_cat = t_cat_0.item()
            
            actual_batch_size = input_img.shape[0]

            if use_jitter:
                t_loc += torch.randn(actual_batch_size, 2)*jitter_stdev

            t_snap = should_snap(t_orient)
            input_img, t_loc, t_orient, t_dims, t_snap = input_img.cuda(), t_loc.cuda(), t_orient.cuda(), t_dims.cuda(), t_snap.cuda()
            d_loc, d_orient = default_loc_orient(actual_batch_size)
            input_img = inverse_xform_img(input_img, t_loc, d_orient.cuda(), img_size)
            t_loc = d_loc.cuda()

            # Update D
            d_loss = 0.0

            # Update G
            g_loss = 0.0

            # Update E + G (VAE step)
            recon_loss = 0.0
            kld_loss = 0.0
            model.set_requires_grad('VAE', t_cat)
            optimizers.g_optimizer(t_cat).zero_grad()
            optimizers.e_optimizer(t_cat).zero_grad()
            real_sdf = render_oriented_sdf((img_size, img_size), t_dims, t_loc, t_orient)
            # mu, logvar = model.encode(real_sdf, input_img, t_cat)
            mu, logvar = model.encode(t_orient, t_cat)
            kld_loss = unitnormal_normal_kld(mu, logvar)
            z = model.sample(mu, logvar)
            #################
            # fake_orient = model.generate(z, input_img, t_cat)
            # recon_loss = F.l1_loss(fake_orient, t_orient)
            # recon_loss = F.l1_loss(fake_orient, t_orient)
            #################
            fake_x, fake_ysign_p = model.generate(z, input_img, t_cat)
            real_x = t_orient[:, 0]
            real_ysign = (t_orient[:, 1] >= 0).float()
            x_recon_loss = F.l1_loss(fake_x.squeeze(), real_x)
            y_recon_loss = F.binary_cross_entropy(fake_ysign_p, real_ysign)
            recon_loss = x_recon_loss + y_recon_loss
            #################
            vae_loss = recon_loss + kld_loss
            vae_loss.backward()
            optimizers.g_optimizer(t_cat).step()
            optimizers.e_optimizer(t_cat).step()

            # Update snap predictor loss
            s_loss = 0.0
            model.set_requires_grad('snap', t_cat)
            model.snapping = True
            optimizers.s_optimizer(t_cat).zero_grad()
            prob = model.snap_predict(input_img, t_cat)
            s_loss = F.binary_cross_entropy(prob, t_snap)
            s_loss.backward()
            optimizers.s_optimizer(t_cat).step()

            if i % log_every == 0:
                catname = categories[t_cat]
                LOG(f'Batch {i}: cat: {catname} | D: {d_loss:4.4} | G: {g_loss:4.4} | Recon: {recon_loss:4.4} | KLD: {kld_loss:4.4} | Snap: {s_loss:4.4}')
        if e % save_every == 0:
            validate()
            model.save(f'{outdir}/model_{e}.pt')
            optimizers.save(f'{outdir}/opt_{e}.pt')

    def validate():
        LOG('Validating')
        model.eval()
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_s_loss = 0.0
        num_batches = 0

        valid_dataset.prepare_same_category_batches(batch_size)
        for i, (input_img, output_mask, t_cat, t_loc, t_orient, t_dims, catcount) in enumerate(valid_loader):
            t_cat = torch.squeeze(t_cat)
            # Verify that we've got only one category in this batch
            t_cat_0 = t_cat[0]
            assert((t_cat == t_cat_0).all())
            t_cat = t_cat_0.item()
            
            actual_batch_size = input_img.shape[0]

            if use_jitter:
                t_loc += torch.randn(actual_batch_size, 2)*jitter_stdev

            t_snap = should_snap(t_orient)
            input_img, t_loc, t_orient, t_dims, t_snap = input_img.cuda(), t_loc.cuda(), t_orient.cuda(), t_dims.cuda(), t_snap.cuda()
            d_loc, d_orient = default_loc_orient(actual_batch_size)
            input_img = inverse_xform_img(input_img, t_loc, d_orient.cuda(), img_size)
            t_loc = d_loc.cuda()

            # VAE losses
            recon_loss = 0.0
            kld_loss = 0.0
            real_sdf = render_oriented_sdf((img_size, img_size), t_dims, t_loc, t_orient)
            # mu, logvar = model.encode(real_sdf, input_img, t_cat)
            mu, logvar = model.encode(t_orient, t_cat)
            kld_loss = unitnormal_normal_kld(mu, logvar)
            z = model.sample(mu, logvar)
            #################
            # fake_orient = model.generate(z, input_img, t_cat)
            # recon_loss = F.l1_loss(fake_orient, t_orient)
            # recon_loss = F.l1_loss(fake_orient, t_orient)
            #################
            fake_x, fake_ysign_p = model.generate(z, input_img, t_cat)
            real_x = t_orient[:, 0]
            real_ysign = (t_orient[:, 1] >= 0).float()
            x_recon_loss = F.l1_loss(fake_x.squeeze(), real_x)
            y_recon_loss = F.binary_cross_entropy(fake_ysign_p, real_ysign)
            recon_loss = x_recon_loss + y_recon_loss
            #################

            # Snap predictor loss
            s_loss = 0.0
            prob = model.snap_predict(input_img, t_cat)
            s_loss = F.binary_cross_entropy(prob, t_snap)

            total_recon_loss += recon_loss
            total_kl_loss += kld_loss
            total_s_loss += s_loss
            num_batches += 1

        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_s_loss = total_s_loss / num_batches
        LOG(f'Recon: {avg_recon_loss:4.4} | KLD: {avg_kl_loss:4.4} | Snap: {avg_s_loss:4.4}')
        model.train()


    for e in range(num_epochs):
        train(e)


    print('FINISHED TRAINING; NOW GENERATING TEST RESULTS...')
    model.eval()
    model.testing = True
    model.snapping = True
    if num_epochs == 0:
        model.load(f'{outdir}/model_{which_to_load}.pt')

    os.system(f'rm -f {outdir}/*.png')

    from PIL import Image

    def tensor2img(x):
        return torchvision.transforms.ToPILImage()(x)

    def tint(x, color):
        colored = torch.zeros(3, x.shape[0], x.shape[1])
        colored[0] = x * (color[0]/255.0)
        colored[1] = x * (color[1]/255.0)
        colored[2] = x * (color[2]/255.0)
        return colored

    def composite_mask(img, mask, color):
        colored = tint(mask, color)
        color_mask_img = tensor2img(colored)
        alpha_mask_img = tensor2img((1.0 - mask).unsqueeze(0))
        return Image.composite(img, color_mask_img, alpha_mask_img)

    test_num_to_gen = 32
    test_dataset.prepare_same_category_batches(batch_size)
    for i, (input_img, output_mask, t_cat, t_loc, t_orient, t_dims, catcount) in enumerate(test_loader):
        # DataLoader sometimes throws an error if the program terminates before iteration over all
        #    data is complete. So, rather than break after N iterations, we do all the iterations
        #    but only do actual work on the first N
        if i < test_num_to_gen:
            print(f'    Generating result {i}/{test_num_to_gen}...')
            t_cat = torch.squeeze(t_cat)

            # Verify that we've got only one category in this batch
            t_cat_0 = t_cat[0]
            assert((t_cat == t_cat_0).all())
            t_cat = t_cat_0.item()

            actual_batch_size = input_img.shape[0]
            input_img, t_loc, t_orient, t_dims = input_img.cuda(), t_loc.cuda(), t_orient.cuda(), t_dims.cuda()
            d_loc, d_orient = default_loc_orient(actual_batch_size)
            input_img = inverse_xform_img(input_img, t_loc, d_orient.cuda(), img_size)
            t_loc = d_loc.cuda()
            noise = torch.randn(actual_batch_size, latent_size)
            noise = noise.cuda()
            with torch.no_grad():
                fake_orient = model.generate(noise, input_img, t_cat)

            walls = input_img.cpu()[:, 3, :, :]

            fake_sdf = render_obb_sdf((img_size, img_size), t_dims, t_loc, fake_orient).cpu()
            fake_mask = (fake_sdf < 0).squeeze().float()
            real_sdf = render_obb_sdf((img_size, img_size), t_dims, t_loc, t_orient).cpu()
            real_mask = (real_sdf < 0).squeeze().float()

            fake_orient_df = render_orientation_sdf((img_size, img_size), t_dims, t_loc, fake_orient).cpu()
            fake_orient_mask_front = (fake_orient_df >= 0).squeeze().float()
            fake_orient_mask_back = (fake_orient_df < 0).squeeze().float()
            real_orient_df = render_orientation_sdf((img_size, img_size), t_dims, t_loc, t_orient).cpu()
            real_orient_mask_front = (real_orient_df >= 0).squeeze().float()
            real_orient_mask_back = (real_orient_df < 0).squeeze().float()

            walls = walls[0]
            real_mask_front = real_mask[0] * real_orient_mask_front[0]
            real_mask_back = real_mask[0] * real_orient_mask_back[0]
            fake_mask_front = fake_mask[0] * fake_orient_mask_front[0]
            fake_mask_back = fake_mask[0] * fake_orient_mask_back[0]

            catname = categories[t_cat]

            # Render real image
            img = tensor2img(tint(walls, (255, 255, 255)))
            img = composite_mask(img, real_mask_front, (255, 100, 100))
            img = composite_mask(img, real_mask_back, (255, 0, 0))
            img.save(f'{outdir}/{i}_REAL_{catname}.png')

            # Render fake image
            img = tensor2img(tint(walls, (255, 255, 255)))
            img = composite_mask(img, fake_mask_front, (100, 100, 255))
            img = composite_mask(img, fake_mask_back, (0, 0, 255))
            img.save(f'{outdir}/{i}_FAKE_{catname}.png')

        elif i == test_num_to_gen:
            print('DONE WITH RESULTS')
