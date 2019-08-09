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
        self.latent_size = latent_size

        def make_encoder():
            return nn.Sequential(
                # 64 -> 32
                DownConvBlock(num_input_channels+1, 8),
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
                nn.Linear(64, 2*latent_size)
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
        
        def make_generator():
            return nn.Sequential(
                nn.Linear(2*latent_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, output_size),
                nn.Softplus()
            )

        def make_discriminator():
            return nn.Sequential(
                # 64 -> 32
                DownConvBlock(num_input_channels+1, 8),
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
        self.generators = nn.ModuleDict()
        self.discriminators = nn.ModuleDict()

        self.encoder = self.make_net_fn(self.encoders, make_encoder)
        self.cond_prior = self.make_net_fn(self.cond_priors, make_cond_prior)
        self.generator = self.make_net_fn(self.generators, make_generator)
        self.discriminator = self.make_net_fn(self.discriminators, make_discriminator)

    def encode(self, sdf, walls, cat):
        mu_logvar = self.encoder(cat)(torch.cat([sdf, walls], dim=1))
        return torch.split(mu_logvar, self.latent_size, dim=1)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        gdis = torch.distributions.Normal(mu, std)
        return gdis.rsample()

    def generate(self, noise, walls, cat):
        enc_walls = self.cond_prior(cat)(walls)
        return self.generator(cat)(torch.cat([noise, enc_walls], dim=1))

    def discriminate(self, sdf, walls, cat):
        return self.discriminator(cat)(torch.cat([sdf, walls], dim=1))

    def set_requires_grad(self, phase, cat):
        if phase == 'D':
            set_requires_grad(self.generator(cat), False)
            set_requires_grad(self.discriminator(cat), True)
            set_requires_grad(self.encoder(cat), False)
            set_requires_grad(self.cond_prior(cat), False)
        elif phase == 'G':
            set_requires_grad(self.generator(cat), True)
            set_requires_grad(self.discriminator(cat), False)
            set_requires_grad(self.encoder(cat), False)
            set_requires_grad(self.cond_prior(cat), True)
        elif phase == 'VAE':
            set_requires_grad(self.generator(cat), True)
            set_requires_grad(self.discriminator(cat), False)
            set_requires_grad(self.encoder(cat), True)
            set_requires_grad(self.cond_prior(cat), True)
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
        self.g_optimizer = self.make_optimizer_fn(self.g_optimizers, [model.generator, model.cond_prior])
        self.d_optimizer = self.make_optimizer_fn(self.d_optimizers, [model.discriminator])
        self.e_optimizer = self.make_optimizer_fn(self.e_optimizers, [model.encoder])

    def save(self, filename):
        g_state = {cat : opt.state_dict() for cat, opt in self.g_optimizers.items()}
        d_state = {cat : opt.state_dict() for cat, opt in self.d_optimizers.items()}
        e_state = {cat : opt.state_dict() for cat, opt in self.e_optimizers.items()}
        torch.save([g_state, d_state, e_state], filename)

    def load(self, filename):
        states = torch.load(filename)
        g_state = states[0]
        d_state = states[1]
        e_state = states[2]
        for cat,state in g_state:
            self.g_optimizer(cat).load_state_dict(state)
        for cat,state in d_state:
            self.d_optimizer(cat).load_state_dict(state)
        for cat,state in e_state:
            self.e_optimizer(cat).load_state_dict(state)

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
                t_loc += torch.randn(actual_batch_size, 2)*jitter_loc_stdev
                shsn = should_snap(t_orient)
                t_orient = torch.where(shsn == torch.zeros_like(shsn), F.normalize(t_orient + torch.randn(actual_batch_size, 2)*jitter_orient_stdev), t_orient)

            input_img, t_loc, t_orient, t_dims = input_img.cuda(), t_loc.cuda(), t_orient.cuda(), t_dims.cuda()
            input_img = inverse_xform_img(input_img, t_loc, t_orient, img_size)
            t_loc, t_orient = default_loc_orient(actual_batch_size)

            real_sdf = render_obb_sdf((img_size, img_size), t_dims, t_loc, t_orient)

            # Update D
            d_loss = 0.0
            model.set_requires_grad('D', t_cat)
            optimizers.d_optimizer(t_cat).zero_grad()
            noise = torch.randn(actual_batch_size, latent_size)
            noise = noise.cuda()
            fake_dims = model.generate(noise, input_img, t_cat).detach()
            d_out_fake = model.discriminate(render_obb_sdf((img_size, img_size), fake_dims, t_loc, t_orient), input_img, t_cat)
            d_out_real = model.discriminate(real_sdf, input_img, t_cat)
            d_loss = 0.5 * torch.mean(-torch.log(d_out_real) - torch.log(1.0 - d_out_fake))
            # for j in range(len(d_out_fake)):
            #     d_out_fake_j = d_out_fake[j]
            #     d_out_real_j = d_out_real[j]
            #     d_loss += 0.5 * torch.mean(-torch.log(d_out_real_j) - torch.log(1.0 - d_out_fake_j))
            # d_loss /= len(d_out_fake)
            d_loss.backward()
            optimizers.d_optimizer(t_cat).step()

            # Update G
            g_loss = 0.0
            model.set_requires_grad('G', t_cat)
            optimizers.g_optimizer(t_cat).zero_grad()
            noise = torch.randn(actual_batch_size, latent_size)
            noise = noise.cuda()
            fake_dims = model.generate(noise, input_img, t_cat)
            d_out_fake = model.discriminate(render_obb_sdf((img_size, img_size), fake_dims, t_loc, t_orient), input_img, t_cat)
            g_loss = torch.mean(-torch.log(d_out_fake))
            # for j in range(len(d_out_fake)):
            #     d_out_fake_j = d_out_fake[j]
            #     g_loss = torch.mean(-torch.log(d_out_fake_j))
            # g_loss /= len(d_out_fake)
            g_loss.backward()
            optimizers.g_optimizer(t_cat).step()

            # Update E + G (VAE step)
            recon_loss = 0.0
            kld_loss = 0.0
            model.set_requires_grad('VAE', t_cat)
            optimizers.g_optimizer(t_cat).zero_grad()
            optimizers.e_optimizer(t_cat).zero_grad()
            mu, logvar = model.encode(real_sdf, input_img, t_cat)
            z = model.sample(mu, logvar)
            fake_dims = model.generate(z, input_img, t_cat)
            recon_loss = F.l1_loss(fake_dims, t_dims)
            kld_loss = unitnormal_normal_kld(mu, logvar)
            vae_loss = recon_loss + kld_loss
            vae_loss.backward()
            optimizers.g_optimizer(t_cat).step()
            optimizers.e_optimizer(t_cat).step()

            if i % log_every == 0:
                catname = categories[t_cat]
                LOG(f'Batch {i}: cat: {catname} | D: {d_loss:4.4} | G: {g_loss:4.4} | Recon: {recon_loss:4.4} | KLD: {kld_loss:4.4}')
        if e % save_every == 0:
            validate()
            model.save(f'{outdir}/model_{e}.pt')
            optimizers.save(f'{outdir}/opt_{e}.pt')

    def validate():
        LOG('Validating')
        model.eval()
        total_recon_loss = 0.0
        total_kl_loss = 0.0
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
                t_loc += torch.randn(actual_batch_size, 2)*jitter_loc_stdev
                shsn = should_snap(t_orient)
                t_orient = torch.where(shsn == torch.zeros_like(shsn), F.normalize(t_orient + torch.randn(actual_batch_size, 2)*jitter_orient_stdev), t_orient)

            input_img, t_loc, t_orient, t_dims = input_img.cuda(), t_loc.cuda(), t_orient.cuda(), t_dims.cuda()
            input_img = inverse_xform_img(input_img, t_loc, t_orient, img_size)
            t_loc, t_orient = default_loc_orient(actual_batch_size)

            real_sdf = render_obb_sdf((img_size, img_size), t_dims, t_loc, t_orient)

            # VAE
            recon_loss = 0.0
            kld_loss = 0.0
            mu, logvar = model.encode(real_sdf, input_img, t_cat)
            z = model.sample(mu, logvar)
            fake_dims = model.generate(z, input_img, t_cat)
            recon_loss = F.l1_loss(fake_dims, t_dims)
            kld_loss = unitnormal_normal_kld(mu, logvar)

            total_recon_loss += recon_loss
            total_kl_loss += kld_loss
            num_batches += 1

        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        LOG(f'Recon: {avg_recon_loss:4.4} | KLD: {avg_kl_loss:4.4}')
        model.train()

    for e in range(num_epochs):
        train(e)

    print('FINISHED TRAINING; NOW GENERATING TEST RESULTS...')
    model.eval()
    if num_epochs == 0:
        model.load(f'{outdir}/model_{which_to_load}.pt')

    os.system(f'rm -f {outdir}/*.png')

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
            input_img = inverse_xform_img(input_img, t_loc, t_orient, img_size)
            t_loc, t_orient = default_loc_orient(actual_batch_size)
            noise = torch.randn(actual_batch_size, latent_size)
            noise = noise.cuda()
            with torch.no_grad():
                fake_dims = model.generate(noise, input_img, t_cat)
            fake_sdf = render_obb_sdf((img_size, img_size), fake_dims, t_loc, t_orient)
            fake_sdf = fake_sdf.cpu()
            fake_mask = (fake_sdf < 0).squeeze()
            real_sdf = render_obb_sdf((img_size, img_size), t_dims, t_loc, t_orient)
            real_sdf = real_sdf.cpu()
            real_mask = (real_sdf < 0).squeeze()
            walls = input_img.cpu()[:, 3, :, :]
            composite = torch.zeros(walls.shape[0], 3, img_size, img_size)
            r_chan = composite[:, 0, :, :]
            g_chan = composite[:, 1, :, :]
            b_chan = composite[:, 2, :, :]
            g_chan = torch.where(fake_mask | real_mask, torch.zeros_like(fake_mask | real_mask).float(), walls)
            r_chan = real_mask.float()
            b_chan = fake_mask.float()
            composite[:, 0, :, :] = r_chan
            composite[:, 1, :, :] = g_chan
            composite[:, 2, :, :] = b_chan

            img = torchvision.transforms.ToPILImage()(composite[0])
            catname = categories[t_cat]
            img.save(f'{outdir}/{i}_{catname}.png')
        elif i == test_num_to_gen:
            print('DONE WITH RESULTS')
