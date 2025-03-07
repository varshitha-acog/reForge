import logging
import os
import sys
import numpy as np
from reforge import io, mdm
from reforge.mdsystem.mdsystem import MDSystem
import reforge.plotting as rfplot
from reforge.utils import logger
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 


logger.setLevel(logging.DEBUG)


def response_1(mat, norm=1):
    resp = np.average(mat, axis=1)
    resp = np.sqrt(resp**2)
    resp = resp.reshape((len(resp) // 3, 3))
    resp = np.sum(resp, axis=1)
    resp /= norm
    return resp


def response_2(mat, norm=1):
    resp = np.average(mat**2, axis=1)
    resp = np.sqrt(resp)
    resp = resp.reshape((len(resp) // 3, 3))
    resp = np.sum(resp, axis=1)
    resp /= norm
    return resp


def response_2_2d(mat, norm=1):
    resp = mat**2
    resp = resp.reshape((resp.shape[0] // 3, resp.shape[1] // 3, 3, 3))
    resp = np.sum(resp, axis=(2, 3))
    resp = np.sqrt(resp)
    resp /= norm
    return resp


def response_force(mat_t):
    nx = mat_t.shape[0]
    ny = mat_t.shape[1]
    nt = mat_t.shape[2]
    t = np.arange(nt)
    k = 0.01
    t = np.sin(2 * np.pi * k * t)
    dt = 2 * np.pi * k * np.cos(2 * np.pi * k * t)
    f = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    f = np.tile(f, (nx // 3, ny // 3))
    force = np.einsum("ij,k->ijk", f, t)
    dforce = np.einsum("ij,k->ijk", f, dt)
    conv = mdm.gfft_conv(dforce, mat_t)
    mat_0 = mat_t[:, :, 0][:, :, None]
    force_0 = force[:, :, 0][:, :, None]
    resp = mat_0 * force - mat_t * force_0 + conv
    # resp  = mat_0 - mat_t
    return resp


def response(mat, norm=1):
    return response_2_2d(mat, norm)


def make_1d_data(infile, nframes=1000):
    print(f"Processing {infile}", file=sys.stderr)
    arrays = []
    mat_t = np.load(infile)
    mat_t = np.swapaxes(mat_t, 0, 1)
    # mat_t = response(mat_t)
    # mat_t -= np.average(mat_t, axis=-1, keepdims=True)
    mat_0 = mat_t[:, :, 0]
    resp_0 = response(mat_0, 1)
    norm = np.average(resp_0)
    if nframes > mat_t.shape[2]:  # Plot only the valid part
        nframes = mat_t.shape[2]
    # arrays.append(np.zeros(pertmat_0.shape[0]))
    for i in range(0, nframes):
        mat = mat_t[:, :, i]
        resp = response(mat, 1)
        arrays.append(resp)
    # arrays /= norm
    print("Finished computing arrays", file=sys.stderr)
    return arrays


def make_2d_data(infile, nframes=1000):
    print(f"Processing {infile}", file=sys.stderr)
    matrices = []
    mat_t = np.load(infile)
    print(mat_t.shape)
    if nframes > mat_t.shape[2]:  # Plot only the valid part
        nframes = mat_t.shape[2]
    print('Checkpoint')
    pertmat_t = mdm.td_perturbation_matrix(mat_t)  
    print(pertmat_t.shape)
    for i in range(0, nframes):
        resp = pertmat_t[:, :, i]
        matrices.append(resp)
    print("Finished computing matrices", file=sys.stderr)
    return matrices


def animate_1d(fig, ax, lines, datas, outfile="data/ani1d.mp4", dt=0.2):
    print("Working on animation", file=sys.stderr)

    def update(frame):
        for line, data in zip(lines, datas):
            line.set_ydata(data[frame])  # Update y-values for each frame
            ax.set_title(f"Time {dt * frame:.2f}, ns")
        return tuple(lines)

    ani = animation.FuncAnimation(
        fig, update, frames=len(datas[0]), interval=50, blit=False
    )
    ani.save(outfile, writer="ffmpeg")  # Save as mp4
    print("Done!", file=sys.stderr)


def animate_2d(fig, img, data, title, dt=0.02, outfile="data/ani2d.mp4"):
    print("Working on animation", file=sys.stderr)

    def update(frame):
        img.set_array(data[frame])
        # img.axes.set_title(f"Time {dt * frame:.2f} ns")
        fig.suptitle(f"{title} :{dt * frame:.2f} ns", fontsize=16)
        return img

    ani = animation.FuncAnimation(
        fig, update, frames=len(data), interval=50, blit=False
    )
    ani.save(outfile, writer="ffmpeg")  # Save as mp4
    print("Done!", file=sys.stderr)


def make_1d_animation(sysdir, sysnames):
    print("Plotting", file=sys.stderr)
    datas = []
    for n, sysname in enumerate(sysnames):
        system = gmxSystem(sysdir, sysname)
        infile = os.path.join(system.datdir, f"corr_pp_slow.npy")
        data = make_2d_data(infile, nframes=2000)
        np.save(f"data/arr_{n}.npy", data)
        datas.append(data)
    # datas = [np.load('data/arr_0.npy'), np.load('data/arr_1.npy'),]
    averages = [np.average(data[0]) for data in datas]
    av = np.average(averages)
    datas = [data / av for data in datas]
    outfile = os.path.join("png", f'pp_{"_".join(sysnames)}.mp4')
    fig, ax, lines = make_plot_t_2d(datas, sysnames, outfile="png/test.png")
    animate_1d(fig, ax, lines, datas, outfile, dt=0.04)


def plot_dci(data, mdsys):
    fig, ax = rfplot.init_figure(grid=(1, 1), axsize=(12, 12))
    img = rfplot.make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=0, vmax=2)
    rfplot.set_hm_parameters(ax, mdsys, xlabel='Perturbed Residue', ylabel='Coupled Residue')
    return fig, img


def make_2d_animation(mdsys):
    logger.info("Animating")
    infile = mdsys.datdir / "corr_pp.npy"
    data = make_2d_data(infile, nframes=1000)
    np.save('data/tmp.npy', data)
    data = np.load('data/tmp.npy')
    outfile = mdsys.pngdir / 'pp_corr.mp4'
    fig, img = plot_dci(data[0], mdsys)
    title = 'RMS Pos-Pos CCF'
    animate_2d(fig, img, data, title, dt=2.0, outfile=outfile)


def make_nm_animation(coords, eval, evec, mode_index, outfile='png/nma.mp4'):
    displacements = evec.reshape(len(evec) // 3, 3) * eval
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c='blue', s=1)
    # Optionally, set equal aspect ratio for 3D axes (a simple way)
    max_range = np.ptp(coords, axis=0).max()
    xmid, ymid, zmid = np.mean(coords, axis=0)
    ax.set_xlim(xmid - max_range/2, xmid + max_range/2)
    ax.set_ylim(ymid - max_range/2, ymid + max_range/2)
    ax.set_zlim(zmid - max_range/2, zmid + max_range/2)
    def update(frame):
        scale = np.sin(frame * 0.5)
        new_coords = coords + scale * displacements
        scat._offsets3d = (new_coords[:, 0], new_coords[:, 1], new_coords[:, 2])
        ax.view_init(elev=30, azim=50)
        ax.dist = 5
        ax.set_title(f"Mode {mode_index} | Scale {scale:.2f}")
        return scat,
    ani = animation.FuncAnimation(fig, update, frames=80, interval=50, blit=False)
    ani.save(outfile, writer="ffmpeg")


if __name__ == '__main__':
    sysdir = 'systems' 
    sysname = 'egfr'
    mdsys = MDSystem(sysdir, sysname)
    make_2d_animation(mdsys)
