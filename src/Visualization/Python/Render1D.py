#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import h5py
import argparse
import sys
import os
import numpy as np
import logging
import matplotlib as mpl


def find_extrema_over_data_set(arr):
    '''
    Find max and min over a range of number arrays
    :param arr: the array over which to find the max and min
    '''

    return (np.nanmin(arr), np.nanmax(arr))


def parse_cmd_line():
    '''
    parse command-line arguments
    :return: dictionary of the command-line args, dashes are underscores
    '''

    parser = argparse.ArgumentParser(
        description='Render 1-dimensional data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group_files = parser.add_mutually_exclusive_group(required=True)
    group_files.add_argument(
        '--file-prefix',
        type=str,
        help="Common prefix of all h5 files being used "
        "in the case of a single h5 file, it is the file name "
        "without the .h5 extension")
    group_files.add_argument('--filename-list',
                             type=str,
                             nargs='+',
                             help="List of all files if they do not "
                             "have same file prefix. You must include '.h5' "
                             "extension ")
    parser.add_argument('--subfile-name',
                        type=str,
                        required=True,
                        help="Name of subfile within h5 file containing "
                        "volume data to be rendered. You must include the "
                        "'.vol' suffix.")
    group_vars = parser.add_mutually_exclusive_group(required=True)
    group_vars.add_argument('--var',
                            type=str,
                            help="Name of variable to render. E.g. 'Psi' "
                            "or 'Error(Psi)'")
    group_vars.add_argument(
        '--list-vars',
        action='store_true',
        help="Print to screen variables in h5 file and exit")
    parser.add_argument(
        '--time',
        type=int,
        required=False,
        help="If specified, renders the integer observation step "
        "instead of an animation")
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        required=False,
                        help="Set the name of the output file you want "
                        "written. For animations this saves an mp4 file and "
                        "for stills a pdf. Name of file should not include "
                        "file extension.")
    group_anim = parser.add_mutually_exclusive_group(required=False)
    group_anim.add_argument(
        '--fps',
        type=float,
        default=5,
        help="Set the number of frames per second when writing "
        "an animation to disk.")
    group_anim.add_argument('--interval',
                            type=float,
                            help="Delay between frames in  milliseconds")
    args = parser.parse_args()
    # Print error message if '.h5' suffix is used in '--file-prefix' argument
    assert args.file_prefix is None or not args.file_prefix.endswith('.h5'),\
    "Retry without .h5 extension on 'file-prefix'"

    return vars(args)


def get_h5_files(files):
    '''
    Get a list of the h5 files containing the data
    :param: list of h5 filenames or common file prefix
    :return: list of h5 files containing volume data
    '''

    h5_files = []
    if isinstance(files, str):
        h5_file_names = glob.glob(files + "[0-9]*.h5")
        assert len(h5_file_names) > 0,\
            "Found no files with prefix {} in directory {}"\
            .format(files, os.getcwd())
        h5_files = [h5py.File(file_name, 'r') for file_name in h5_file_names]

    else:
        for file_name in files:
            try:
                h5_files.append(h5py.File(file_name, 'r'))
            except IOError as err:
                sys.exit("Could not find file '{}' in directory {}: {}".format(
                    file_name, os.getcwd(), err))
    return h5_files


def print_var_names(files, subfile_name):
    '''
    Print all available variables to screen
    :param files: list of h5 filenames or common file prefix
    :param subfile_name: name of .vol subfile in h5 file(s)
    :return: None
    '''

    h5files = get_h5_files(files)
    volfile = h5files[0][subfile_name]
    obs_id_0 = next(iter(volfile))
    variables = list(volfile[obs_id_0].keys())
    variables.remove("connectivity")
    variables.remove("InertialCoordinates_x")
    variables.remove("total_extents")
    variables.remove("grid_names")
    variables.remove("bases")
    variables.remove("quadratures")
    print("Variables in H5 file:\n{}".format(list(map(str, variables))))
    for h5_file in h5files:
        h5_file.close()


def get_data(files, subfile_name, var_name):
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
    h5files = get_h5_files(files)
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
    ax = plt.axes(xlim=(find_extrema_over_data_set(coords)),
                  ylim=(find_extrema_over_data_set(data)))
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_cmd_line()
    if args['output'] is not None:
        mpl.use("Agg")
    if args['file_prefix'] is not None:
        files = args['file_prefix']
    else:
        files = args['filename_list']
    subfile_name = args['subfile_name']
    if args['list_vars']:
        print_var_names(files, subfile_name)
        sys.exit(0)
    time, coords, data = get_data(files, subfile_name, args['var'])
    if args['interval'] is None:
        interval = 1000.0 / args['fps']
    else:
        interval = args['interval']
    if args['time'] is None:
        render_animation(args['var'], args['output'], interval, time, coords,
                         data)
    else:
        render_single_time(args['var'], args['time'], args['output'], time,
                           coords, data)
