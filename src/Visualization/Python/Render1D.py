#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import h5py
import sys
import os
import numpy as np
import logging
import matplotlib as mpl
import click
import rich
from spectre.Visualization.ReadH5 import available_subfiles


def find_extrema_over_data_set(arr):
    '''
    Find max and min over a range of number arrays
    :param arr: the array over which to find the max and min
    '''

    return (np.nanmin(arr), np.nanmax(arr))


def get_data(h5files, subfile_name, var_name):
    '''
    Get the data to be plotted
    :param files: list of h5 filenames or common file prefix
    :param subfile_name: name of .vol subfile in h5 file(s)
    :param var_name: name of variable to render
    :return: the list of time, coords and data
    '''

    time = []
    coords = []
    data = []
    volfiles = [h5file[subfile_name] for h5file in h5files]
    # Get a list of times from the first vol file
    ids_times = [(obs_id, volfiles[0][obs_id].attrs['observation_value'])
                 for obs_id in volfiles[0].keys()]
    ids_times.sort(key=lambda pair: pair[1])
    for obs_id, local_time in ids_times:
        local_coords = []
        local_data = []
        for volfile in volfiles:
            try:
                local_data = (local_data + list(volfile[obs_id][var_name]))
            except KeyError:
                print("The variable name {} is not a valid variable. "
                      "Use '--list-vars' to see the list of variable names in "
                      "the file(s) \n{}".format(var_name,
                                                list(map(str, h5files))))
                sys.exit(1)
            local_coords = (local_coords +
                            list(volfile[obs_id]['InertialCoordinates_x']))
        ordering = np.argsort(local_coords)
        coords.append(np.array(local_coords)[ordering])
        data.append(np.array(local_data)[ordering])
        time.append(local_time)

    for h5file in h5files:
        h5file.close()
    return time, coords, data


def render_single_time(var_name, time_slice, output_prefix, time, coords,
                       data):
    '''
    Renders image at a single time step
    :param var_name: name of variable to render
    :param time_slice: the integer observation step to render
    :param output_prefix: name of output file
    :param time: list of time steps
    :param coords: list of coordinates to plot
    :param data: list of variable data to plot
    :return: None
    '''
    import matplotlib.pyplot as plt

    plt.xlabel("x")
    plt.ylabel(var_name)
    try:
        plt.title("t = {:.5f}".format(time[time_slice]))
        plt.plot(coords[time_slice], data[time_slice], 'o')
    except IndexError:
        sys.exit("The integer time step provided {} is outside the range of "
                 "allowed time steps. The range of allowed integer time steps "
                 "in the files provided is 0-{}".format(
                     time_slice,
                     len(time) - 1))
    if output_prefix:
        logging.info("Writing still to file {}.pdf".format(output_prefix))
        plt.savefig(output_prefix + ".pdf", format='pdf')
    else:
        plt.show()


def render_animation(var_name, output_prefix, interval, time, coords, data):
    '''
    Render an animation of the data
    :param var_name: name of variable to render
    :param output_prefix: name of output file
    :param interval: delay between frames for animation
    :param time: list of time steps
    :param coords: list of coordinates to plot
    :param data: list of variable data to plot
    :return: None
    '''
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    fig = plt.figure()
    ax = plt.axes(xlim=(find_extrema_over_data_set(
        np.concatenate(np.asarray(coords)).ravel())),
                  ylim=(find_extrema_over_data_set(
                      np.concatenate(np.asarray(data)).ravel())))
    line, = ax.plot([], [], 'o', lw=2)
    ax.set_xlabel('x')
    ax.set_ylabel(var_name)
    title = ax.set_title("")

    def init():
        '''
        Initialize the animation canvas
        :return: empty line and title info
        '''
        line.set_data([], [])
        title.set_text("")
        return line, title

    def animate(iteration):
        '''
        Update the animation canvas
        :return: line and title info for particular time step
        '''
        title.set_text("t = {:.5f}".format(time[iteration]))
        line.set_data(coords[iteration], data[iteration])
        return line, title

    anim = animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init,
                                   frames=len(time),
                                   interval=interval)
    if output_prefix:
        fps = 1000.0 / interval
        logging.info(
            "Writing animation to file {}.mp4 at {} frames per second".format(
                output_prefix, fps))
        anim.save(output_prefix + ".mp4", writer='ffmpeg')
    else:
        plt.show()


@click.command()
@click.argument("h5_files",
                nargs=-1,
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                readable=True))
@click.option("--subfile-name",
              "-d",
              help=("Name of subfile within h5 file containing "
                    "1D volume data to be rendered."))
@click.option("--var",
              "-y",
              help=("Name of variable to render. E.g. 'Psi' "
                    "or 'Error(Psi)'. Can be specified multiple times. "
                    "If unspecified, print available variables and exit."))
@click.option("--list-vars",
              "-l",
              is_flag=True,
              help="Print available variables and exit.")
@click.option("--step",
              type=int,
              help=("If specified, renders the integer observation step "
                    "instead of an animation. Set to '-1' for the last step."))
@click.option("-o",
              "--output",
              help=("Set the name of the output file you want "
                    "written. For animations this saves an mp4 file and "
                    "for stills a pdf. Name of file should not include "
                    "file extension."))
@click.option('--fps',
              type=float,
              default=5,
              help=("Set the number of frames per second when writing "
                    "an animation to disk."))
@click.option('--interval',
              type=float,
              help="Delay between frames in milliseconds")
def render_1d_command(h5_files, subfile_name, list_vars, **args):
    """Render 1D data"""
    # Script should be a noop if input files are empty
    if not h5_files:
        return

    open_h5_files = [h5py.File(filename, "r") for filename in h5_files]

    # Print available subfile names and exit
    if not subfile_name:
        import rich.columns
        rich.print(
            rich.columns.Columns(
                available_subfiles(open_h5_files[0], extension=".vol")))
        return

    if not subfile_name.endswith(".vol"):
        subfile_name += ".vol"

    # Print available variables and exit
    if list_vars or not args['var']:
        volfile = open_h5_files[0][subfile_name]
        obs_id_0 = next(iter(volfile))
        variables = list(volfile[obs_id_0].keys())
        variables.remove("connectivity")
        variables.remove("InertialCoordinates_x")
        variables.remove("total_extents")
        variables.remove("grid_names")
        variables.remove("bases")
        variables.remove("quadratures")
        if "domain" in variables:
            variables.remove("domain")
        if "functions_of_time" in variables:
            variables.remove("functions_of_time")

        import rich.columns
        rich.print(rich.columns.Columns(variables))
        return

    time, coords, data = get_data(open_h5_files, subfile_name, args['var'])
    if args['interval'] is None:
        interval = 1000.0 / args['fps']
    else:
        interval = args['interval']
    if args['step'] is None:
        render_animation(args['var'], args['output'], interval, time, coords,
                         data)
    else:
        render_single_time(args['var'], args['step'], args['output'], time,
                           coords, data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    render_1d_command(help_option_name=["-h", "--help"])
