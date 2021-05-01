import math

def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    last_epoch = 0
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step:
            last_epoch = curriculum_step
    return last_epoch

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    # step = get_current_step(curriculum, epoch)
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict

# CARLA DATASET SCRIPT
CARLAFORCARS = {
    0: {'batch_size': 30, 'num_steps': 48, 'img_size': 32, 'batch_split': 1, 'gen_lr': 4e-5, 'disc_lr': 4e-4},
    int(30e3): {'batch_size': 16, 'num_steps': 48, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(75e3): {'batch_size': 16, 'num_steps': 48, 'img_size': 128, 'batch_split': 8, 'gen_lr': 10e-6,
                'disc_lr': 10e-5},
    # int(200e3): {},

    'fov': 30,
    'ray_start': 0.75,
    'ray_end': 1.25,
    'fade_steps': 10000,
    'sample_dist': 'uniform',
    'h_stddev': math.pi,
    'v_stddev': math.pi / 4 * 85 / 90,
    'h_mean': math.pi * 0.5,
    'v_mean': math.pi / 4 * 85 / 90,
    'topk_interval': 1000,
    'topk_v': 0.5,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 1,
    'model': 'TALLSIREN',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'CARLA',
    'white_back': True,
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': False,
    'z_lambda': 0,
    'pos_lambda': 0,
    'learnable_dist': False,
}

CARLAFORSHAPENETCARS = {
    0: {'batch_size': 30, 'num_steps': 48, 'img_size': 32, 'batch_split': 1, 'gen_lr': 4e-5, 'disc_lr': 4e-4},
    int(30e3): {'batch_size': 16, 'num_steps': 48, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(75e3): {'batch_size': 16, 'num_steps': 48, 'img_size': 128, 'batch_split': 8, 'gen_lr': 10e-6,
                'disc_lr': 10e-5},
    int(200e3): {},

    'fov': 30,
    'ray_start': 0.75,
    'ray_end': 1.25,
    'fade_steps': 10000,
    'sample_dist': 'uniform',
    'h_stddev': math.pi,
    'v_stddev': math.pi / 4 * 85 / 90,
    'h_mean': math.pi * 0.5,
    'v_mean': math.pi / 4 * 85 / 90,
    'topk_interval': 1000,
    'topk_v': 0.5,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 1,
    'model': 'TALLSIREN',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'ShapenetCars',
    'white_back': True,
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': False,
    'z_lambda': 0,
    'pos_lambda': 0,
    'learnable_dist': False,
}

CARLAFORSHAPENETSHIPS = {
    0: {'batch_size': 30, 'num_steps': 48, 'img_size': 32, 'batch_split': 1, 'gen_lr': 4e-5, 'disc_lr': 4e-4},
    int(30e3): {'batch_size': 16, 'num_steps': 48, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(75e3): {'batch_size': 16, 'num_steps': 48, 'img_size': 128, 'batch_split': 8, 'gen_lr': 10e-6,
                'disc_lr': 10e-5},
    # int(200e3): {},

    'fov': 30,
    'ray_start': 0.75,
    'ray_end': 1.25,
    'fade_steps': 10000,
    'sample_dist': 'uniform',
    'h_stddev': math.pi,
    'v_stddev': math.pi / 4 * 85 / 90,
    'h_mean': math.pi * 0.5,
    'v_mean': math.pi / 4 * 85 / 90,
    'topk_interval': 1000,
    'topk_v': 0.5,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 1,
    'model': 'TALLSIREN',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'ShapenetShips',
    'white_back': True,
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': False,
    'z_lambda': 0,
    'pos_lambda': 0,
    'learnable_dist': False,
}

# Trained on one RTX 8000. If training on two RTX 6000's, halve the batch size.
SPATIALSIRENBASELINELB = {
    0: {'batch_size': 52*2, 'num_steps': 12, 'img_size': 32, 'batch_split': 1, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(30e3): {'batch_size': 28*2, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(70e3): {'batch_size': 28*2, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 1e-5, 'disc_lr': 1e-4},
    int(104e3): {'batch_size': 6*2, 'num_steps': 12, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-6, 'disc_lr': 2e-5},
    int(200e3): {},

    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 0.3,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15.,
    'last_back': False,
}

PEREGRINETESTCURRICULUM = {
    0: {'batch_size': 16, 'num_steps': 48, 'img_size': 32, 'batch_split': 2, 'gen_lr': 4e-5, 'disc_lr': 4e-4},

    'fov': 30,
    'ray_start': 0.75,
    'ray_end': 1.25,
    'fade_steps': 10000,
    'sample_dist': 'uniform',
    'h_stddev': math.pi,
    'v_stddev': math.pi / 4 * 85 / 90,
    'h_mean': math.pi * 0.5,
    'v_mean': math.pi / 4 * 85 / 90,
    'topk_interval': 1000,
    'topk_v': 0.5,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 1,
    'model': 'TALLSIREN',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'CARLA',
    'white_back': True,
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': False,
    'z_lambda': 0,
    'pos_lambda': 0,
    'learnable_dist': False,
}
