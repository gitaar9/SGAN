import torch
from matplotlib import pyplot as plt
import json
import numpy as np
import os
from getpass import getpass


def load_array_from_tb_json(path):
    if '.json' in path:
        with open(path) as json_file:
            data = json.load(json_file)
            data = np.asarray(data)[:, 1:]
    else:
        with open(path) as txt_file:
            lines = [l.strip() for l in txt_file.readlines() if l.strip() != ""]
            data = []
            for l in lines:
                epoch, fid = l.split(':')
                epoch = int(epoch)
                fid = float(fid)
                data.append([epoch, fid])
            data = np.asarray(data)
    return data


def plot_array(a, label=None):
    x = a[:, 0]
    y = a[:, 1]
    plt.plot(x, y, label=label)


def show_plot(title, x_label, y_label, x_axis_limit=None, y_lims=None):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if x_axis_limit is not None:
        plt.xlim((0, x_axis_limit))
    if y_lims is not None:
        plt.ylim(y_lims)


def all_plotting(dataset_names, json_path, *args, **kwargs):
    for ds_name in dataset_names:
        plot_array(load_array_from_tb_json(json_path.format(ds_name)), ds_name)
    show_plot(*args, **kwargs)


# GRAF
def plot_graf_plots():
    carla_path_name = 'carla'
    sncars_path_name = 'shapenetcars'
    snships_path_name = 'shapenetships'
    fid_path = 'new_plan_result/run-{}_128_new_plan_monitoring-tag-validation_fid.json'
    kid_path = 'new_plan_result/run-{}_128_new_plan_monitoring-tag-validation_kid.json'

    all_plotting(
        [carla_path_name, sncars_path_name, snships_path_name],
        # ['shapenetcars_varying_distance'],
        fid_path,
        'Frechet Inception Distance for GRAF',
        'Epoch',
        'FID'
    )

    all_plotting(
        [carla_path_name, sncars_path_name, snships_path_name],
        kid_path,
        'Kernel Inception Distance for GRAF',
        'Epoch',
        'KID'
    )


def plot_simple_torch_array_file(dataset_names, file_path, *args, **kwargs):
    for ds_name in dataset_names:
        losses = torch.load(file_path.format(ds_name))
        plt.plot(list(range(len(losses))), losses, label=ds_name)
    show_plot(*args, **kwargs)


# Pi-GAN
def plot_pi_gan_plots(local_names, *args, **kwargs):
    fid_path = 'mirror_loss_results/sgan_{}_fid.txt'

    plt.axvline(x=30000, color='red', linestyle=':')
    plt.axvline(x=75000, color='red', linestyle=':')
    all_plotting(
        local_names,
        fid_path,
        'Frechet Inception Distance for Pi-GAN',
        'Batch',
        'FID',
        *args,
        **kwargs
    )


def download_files_from_peregrine(password, peregrine_names, local_names, peregrine_path, local_path):
    peregrine_adress = 'sshpass -p "{}" scp s2576597@peregrine.hpc.rug.nl:{} {}'
    for p, l in zip(peregrine_names, local_names):
        os.system(peregrine_adress.format(password, peregrine_path.format(p), local_path.format(l)))


def carla_cars_mirror(password):
    # FID plot
    fid_local_names = ['cars_mirror_loss_density', 'cars_mirror_loss_color', 'cars_mirror']
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=['cars_mirror_loss_density', 'cars_mirror_loss_color', 'cars_mirror'],
            local_names=fid_local_names,
            peregrine_path='/data/s2576597/SGAN/carla_for_{}/fid.txt',
            local_path='mirror_loss_results/sgan_{}_fid.txt'
        )
    plt.figure(0)
    plot_pi_gan_plots(fid_local_names)

    # Sym loss plot
    sym_loss_local_names = ['cars_mirror_loss_density', 'cars_mirror_loss_color']
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=['cars_mirror_loss_density', 'cars_mirror_loss_color'],
            local_names=sym_loss_local_names,
            peregrine_path='/data/s2576597/SGAN/carla_for_{}/generator_sym.losses',
            local_path='mirror_loss_results/sgan_{}_generator_sym.losses'
        )
    plt.figure(1)
    plot_simple_torch_array_file(sym_loss_local_names, 'mirror_loss_results/sgan_{}_generator_sym.losses', 'Sym loss', 'time', 'loss')

    # Gen loss plot
    sym_loss_local_names = ['cars_mirror_loss_density', 'cars_mirror_loss_color']
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=['cars_mirror_loss_density', 'cars_mirror_loss_color'],
            local_names=sym_loss_local_names,
            peregrine_path='/data/s2576597/SGAN/carla_for_{}/generator.losses',
            local_path='mirror_loss_results/sgan_{}_generator.losses'
        )
    plt.figure(2)
    plot_simple_torch_array_file(sym_loss_local_names, 'mirror_loss_results/sgan_{}_generator.losses', 'Gen loss', 'time', 'loss')
    plt.show()


def snships_mirror_l2(password):
    # FID plot
    # old = , 'shapenetcars_sym_loss', 'shapenetcars_baseline'
    local_names = ['shapenetships_baseline', 'shapenetships_sym_loss', 'shapenetships_sym_loss_100_lambda']
    peregrine_names = ['shapenetships_baseline', 'shapenetships_sym_loss', 'shapenetships_sym_loss_100_lambda']

    x_axis_limit = 80000
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/fid.txt',
            local_path='mirror_loss_results/sgan_{}_fid.txt'
        )
    plt.figure(0)
    plot_pi_gan_plots(local_names, x_axis_limit=x_axis_limit, y_lims=(100, 300))

    # Sym loss plot
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/generator_sym.losses',
            local_path='mirror_loss_results/sgan_{}_generator_sym.losses'
        )
    plt.figure(1)
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_generator_sym.losses', 'Symmetry based loss', 'Batch', 'L2 loss', x_axis_limit=x_axis_limit, y_lims=(0, 50))

    # Gen loss plot
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/generator.losses',
            local_path='mirror_loss_results/sgan_{}_generator.losses'
        )
    plt.figure(2)
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_generator.losses', 'Gen loss', 'time', 'loss', x_axis_limit=x_axis_limit, y_lims=(0, 10))

    # Max GPU memory
    local_names = ['shapenetships_sym_loss_100_lambda', 'shapenetships_baseline_hierarchical']
    peregrine_names = ['shapenetships_sym_loss_100_lambda', 'shapenetships_baseline_hierarchical']
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/max_memories.sizes',
            local_path='mirror_loss_results/sgan_{}_max_memories.sizes'
        )
    plt.figure(3)
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_max_memories.sizes', 'Max GPU mem', 'Iter', 'Memory')

    plt.show()


def snships_mirror_l2_hierarchical(password):
    # FID plot
    local_names = ['shapenetships_baseline', 'shapenetships_sym_loss', 'shapenetships_baseline_hierarchical', 'shapenetships_sym_loss_hierarchical']
    peregrine_names = ['shapenetships_baseline', 'shapenetships_sym_loss', 'shapenetships_baseline_hierarchical', 'shapenetships_sym_loss_hierarchical']
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/fid.txt',
            local_path='mirror_loss_results/sgan_{}_fid.txt'
        )
    plt.figure(0)
    plot_pi_gan_plots(local_names)

    # Sym loss plot
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/generator_sym.losses',
            local_path='mirror_loss_results/sgan_{}_generator_sym.losses'
        )
    plt.figure(1)
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_generator_sym.losses', 'Sym loss', 'time', 'loss')

    # Gen loss plot
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/generator.losses',
            local_path='mirror_loss_results/sgan_{}_generator.losses'
        )
    plt.figure(2)
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_generator.losses', 'Gen loss', 'time', 'loss')

    # Max GPU memory
    local_names = ['shapenetships_sym_loss_100_lambda', 'shapenetships_baseline_hierarchical']
    peregrine_names = ['shapenetships_sym_loss_100_lambda', 'shapenetships_baseline_hierarchical']
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/max_memories.sizes',
            local_path='mirror_loss_results/sgan_{}_max_memories.sizes'
        )
    plt.figure(3)
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_max_memories.sizes', 'Max GPU mem', 'time', 'Memory')

    plt.show()


def download_model_files(password, run_name):
    peregrine_adress = 'sshpass -p "{}" scp s2576597@peregrine.hpc.rug.nl:{} {}'
    p_location = '/data/s2576597/SGAN/{}/'.format(run_name)
    l_location = '/samsung_hdd/Files/AI/TNO/S-GAN-prerelease/models/{}/'.format(run_name)
    files = [
        "ema2.pth",
        "ema.pth",
        "optimizer_D.pth",
        "optimizer_G.pth",
        "generator.pth",
        "discriminator.pth",
        "scaler.pth"
    ]
    os.system(f"mkdir {l_location}")
    for file_name in files:
        print(f"Downloading {p_location + file_name}")
        os.system(peregrine_adress.format(password, p_location + file_name, l_location))


def sncar_mirror_l2_hierarchical(password):
    # FID plot
    local_names = ['shapenetcars_no_mirror', 'shapenetcars_baseline_hierarchical', 'shapenetcars_sym_loss_hierarchical']
    peregrine_names = ['shapenetcars_no_mirror', 'shapenetcars_baseline_hierarchical', 'shapenetcars_sym_loss_hierarchical']
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/fid.txt',
            local_path='mirror_loss_results/sgan_{}_fid.txt'
        )
    plt.figure(0)
    plot_pi_gan_plots(local_names)

    # Sym loss plot
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/generator_sym.losses',
            local_path='mirror_loss_results/sgan_{}_generator_sym.losses'
        )
    plt.figure(1)
    plot_simple_torch_array_file(['shapenetcars_baseline_hierarchical', 'shapenetcars_sym_loss_hierarchical'], 'mirror_loss_results/sgan_{}_generator_sym.losses', 'Sym loss', 'time', 'loss')

    # Gen loss plot
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/generator.losses',
            local_path='mirror_loss_results/sgan_{}_generator.losses'
        )
    plt.figure(2)
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_generator.losses', 'Gen loss', 'time', 'loss')

    # Max GPU memory
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/max_memories.sizes',
            local_path='mirror_loss_results/sgan_{}_max_memories.sizes'
        )
    plt.figure(3)
    plot_simple_torch_array_file(['shapenetcars_baseline_hierarchical', 'shapenetcars_sym_loss_hierarchical'], 'mirror_loss_results/sgan_{}_max_memories.sizes', 'Max GPU mem', 'time', 'Memory')

    plt.show()


def new_snship_runs(password):
    # FID plot
    local_names = ['carla_sym_loss_hierarchical_v2', 'shapenetships_sym_loss_hierarchical_v2', 'shapenetships_no_mirror_v2']
    peregrine_names = ['carla_sym_loss_hierarchical_v2', 'shapenetships_sym_loss_hierarchical_v2', 'shapenetships_no_mirror_v2']
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/fid.txt',
            local_path='mirror_loss_results/sgan_{}_fid.txt'
        )
    plt.figure(0)
    plot_pi_gan_plots(local_names)

    # Sym loss plot
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/generator_sym.losses',
            local_path='mirror_loss_results/sgan_{}_generator_sym.losses'
        )
    plt.figure(1)
    plot_simple_torch_array_file(['shapenetcars_baseline_hierarchical', 'shapenetcars_sym_loss_hierarchical'], 'mirror_loss_results/sgan_{}_generator_sym.losses', 'Sym loss', 'time', 'loss')

    # Gen loss plot
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/generator.losses',
            local_path='mirror_loss_results/sgan_{}_generator.losses'
        )
    plt.figure(2)
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_generator.losses', 'Gen loss', 'time', 'loss')

    # Max GPU memory
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/max_memories.sizes',
            local_path='mirror_loss_results/sgan_{}_max_memories.sizes'
        )
    plt.figure(3)
    local_names = local_names[:2]
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_max_memories.sizes', 'Max GPU mem', 'time', 'Memory')

    plt.show()


def final_sncar_runs(password):
    # FID plot
    local_names = ['carla_sym_loss_hierarchical_v2', 'carla_no_mirror_v2', 'shapenetcars_sym_loss_hierarchical_v3', 'shapenetcars_no_mirror_v3']
    peregrine_names = ['carla_sym_loss_hierarchical_v2', 'carla_no_mirror_v2', 'shapenetcars_sym_loss_hierarchical_v3', 'shapenetcars_no_mirror_v3']
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/fid.txt',
            local_path='mirror_loss_results/sgan_{}_fid.txt'
        )
    plt.figure(0)
    plot_pi_gan_plots(local_names)

    # Sym loss plot
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/generator_sym.losses',
            local_path='mirror_loss_results/sgan_{}_generator_sym.losses'
        )
    plt.figure(1)
    plot_simple_torch_array_file(['shapenetcars_baseline_hierarchical', 'shapenetcars_sym_loss_hierarchical'], 'mirror_loss_results/sgan_{}_generator_sym.losses', 'Sym loss', 'time', 'loss')

    # Gen loss plot
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/generator.losses',
            local_path='mirror_loss_results/sgan_{}_generator.losses'
        )
    plt.figure(2)
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_generator.losses', 'Gen loss', 'time', 'loss')

    # Max GPU memory
    if password:
        download_files_from_peregrine(
            password=password,
            peregrine_names=peregrine_names,
            local_names=local_names,
            peregrine_path='/data/s2576597/SGAN/{}/max_memories.sizes',
            local_path='mirror_loss_results/sgan_{}_max_memories.sizes'
        )
    plt.figure(3)
    local_names = [n for n in local_names if 'no_mirror' not in n]
    plot_simple_torch_array_file(local_names, 'mirror_loss_results/sgan_{}_max_memories.sizes', 'Max GPU mem', 'time', 'Memory')

    plt.show()


def main():
    password = getpass()
    final_sncar_runs(password)
    # new_snship_runs(password)
    # snships_mirror_l2_hierarchical(password)
    # sncar_mirror_l2_hierarchical(password)
    # snships_mirror_l2(password)
    # carla_cars_mirror(password)
    # download_model_files(password, 'shapenetcars_sym_loss_hierarchical')


if __name__ == '__main__':
    main()


# scp s2576597@peregrine.hpc.rug.nl:/data/s2576597/SGAN/carla_for_shapenetcars/fid.txt new_plan_result/sgan_shapenetcars_fid.txt
# scp s2576597@peregrine.hpc.rug.nl:/data/s2576597/SGAN/carla_for_shapenetships/fid.txt new_plan_result/sgan_shapenetships_fid.txt
# scp s2576597@peregrine.hpc.rug.nl:/data/s2576597/SGAN/carla_for_cars/fid.txt new_plan_result/sgan_carla_fid.txt