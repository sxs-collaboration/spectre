#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import h5py
import numpy as np

# Set back to normal LaTeX math font...
import matplotlib.style
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

matplotlib.use('pdf')
import matplotlib.pyplot as plt


def read_and_plot_data(dat_file, args):
    """Read the data from the dat file and generate a plot.

    Checks that various preconditions are met and throws
    an error if they are not.
    """
    legend = [(x.decode('ascii') if type(x) == bytes else x)
              for x in dat_file.attrs['Legend']]
    all_data = np.asarray(dat_file)
    if args['x_axis'] == None:
        args['x_axis'] = legend[0]
    elif not args['x_axis'] in legend:
        raise ValueError(
            "Unknown x-axis function '{}'\nKnown functions are:\n{}".format(
                args['x_axis'], str(legend)))

    # make sure all requested functions are in the file:
    for i_function in range(len(args['functions'])):
        function = args['functions'][i_function]
        if not function in legend:
            raise ValueError(
                "Unknown function '{}'\nKnown functions are:\n{}".format(
                    function, str(legend)))

        this_linestyle = args['linestyles'][
            i_function if len(args['linestyles']) > 1 else 0]
        this_label = (dict(args['labels'])[function] if args['labels'] != None
                      and function in args['labels'] else function)

        plt.plot(all_data[:, legend.index(args['x_axis'])],
                 all_data[:, legend.index(function)],
                 label=this_label,
                 linestyle=this_linestyle,
                 linewidth=args['linewidth'])

    if args['y_logscale']:
        plt.yscale('log')
    plt.xlabel(args['x_label'] if args['x_label'] != None else args['x_axis'],
               fontsize=args['fontsize'])
    if args['y_label']:
        plt.ylabel(args['y_label'], fontsize=args['fontsize'])

    _ = plt.legend(fontsize=args['fontsize'] - 1,
                   loc='best',
                   ncol=args['legend_ncols'])
    plt.xticks(fontsize=args['fontsize'])
    plt.yticks(fontsize=args['fontsize'])
    if args['x_bounds']:
        plt.xlim(float(args['x_bounds'][0]), float(args['x_bounds'][1]))
    if args['y_bounds']:
        plt.ylim(float(args['y_bounds'][0]), float(args['y_bounds'][1]))
    if args['title'] != None:
        plt.title(args['title'], fontsize=args['fontsize'])

    print("Saving figure to '{}'".format(args['output'] + '.pdf'))
    plt.savefig(args['output'] + '.pdf',
                transparent=True,
                format='pdf',
                bbox_inches='tight')
    return None


def process_subfile(dat_file, legend_only, write_dat, **kwargs):
    """Given the read in subfile, plot the data.

    If --legend-only was specified then the legend is printed
    and execution terminated.

    If --write-dat is specified then the data is written to a text dat file.

    Generates a plot and writes it to disk using read_and_plot_data.
    """
    legend = [(x.decode('ascii') if type(x) == bytes else x)
              for x in dat_file.attrs['Legend']]
    if legend_only:
        print("The legend in the dat subfile '{}' is:\n{}".format(
            dat_file.name, legend))
        return legend

    if not (write_dat is None):
        print("Writing to text dat file '{}'".format(write_dat))
        np.savetxt(write_dat,
                   np.asarray(dat_file),
                   delimiter=' ',
                   header='\n'.join(
                       '{}' for _ in range(len(legend))).format(*legend))
        return None

    read_and_plot_data(dat_file, kwargs)
    return None


def open_and_process_file(filename, subfile_name, **kwargs):
    """Opens the specified HDF5 file.

    If no subfile is specified this function prints all available Dat
    subfiles in the root of the HDF5 and then exists. If a subfile is
    specified, the function forwards on to process_subfile()
    """
    def available_subfiles(h5file, path='/'):
        subfiles = []
        for member in h5file.keys():
            if type(h5file[member]) is h5py._hl.group.Group:
                subfiles = subfiles + available_subfiles(
                    h5file[member], path + member + '/')
            elif member.endswith('.dat'):
                subfiles.append(path + member)
        return subfiles

    with h5py.File(filename, 'r') as h5file:
        if subfile_name != None:
            dat_file = h5file.get(subfile_name + '.dat')
            if dat_file is None:
                raise Exception(
                    "Unable to open dat file '%s'. Available files are:\n%s" %
                    (subfile_name + '.dat', available_subfiles(h5file)))
            return process_subfile(dat_file, **kwargs)
        else:
            print("No subfile was specified. If you want to plot or write "
                  "data, you need to specify a dat file using the "
                  "--subfile-name option.")
            print("Available dat files are:", available_subfiles(h5file))
            return available_subfiles(h5file)
    return None


def parse_args():
    """
    Parse the command line arguments
    """
    import argparse as ap
    import json

    # We use a custom formatter to allow us to override the default
    # formatting style when we want to add line breaks ourselves.
    class SmartFormatter(ap.HelpFormatter):
        def _split_lines(self, text, width):
            if text.startswith('R|'):
                return text[2:].splitlines()
            # If the user did not ask for raw formatting use argparse's
            # formatter instead.
            return ap.HelpFormatter._split_lines(self, text, width)

    parser = ap.ArgumentParser(
        description=
        "Analyze reduction data files (.dat HDF5 subfiles) written out "
        "by SpECTRE. Plots of the data can be generated, looking at "
        "what variables were written to the subfile, and writing the "
        "dat subfile to a text file on disk.",
        formatter_class=SmartFormatter)
    parser.add_argument('filename', help="The HDF5 file to read.")
    parser.add_argument(
        '--subfile-name',
        '-d',
        required=False,
        help="The dat subfile to read excluding the .dat extension. "
        "If excluded all available dat subfiles will be printed.")

    action_group = parser.add_mutually_exclusive_group(required=True)

    action_group.add_argument('--legend-only',
                              action='store_true',
                              help="If specified, only print out the legend")
    action_group.add_argument(
        '--write-dat',
        help="If specified, writes the entire dat file to a text Dat file with "
        "spaces as the delimiter and exits.")

    action_group.add_argument(
        '--output',
        '-o',
        help="Name of the plot file. The '.pdf' extension will "
        "be added automatically. Note: Requires additional arguments to be "
        "specified. Specify --help followed by '--output' or '-o' to see "
        "the additional arguments.")

    import sys
    if '--output' in sys.argv or '-o' in sys.argv:
        parser.add_argument('--functions',
                            required=True,
                            nargs='+',
                            help="The quantities to plot as a function "
                            "of '--x-axis'.")

        # Specify all plotting options
        parser.add_argument('--x-bounds',
                            required=False,
                            nargs=2,
                            help="The lower and upper bounds of the x-axis.")
        parser.add_argument('--y-bounds',
                            required=False,
                            nargs=2,
                            help="The lower and upper bounds of the y-axis.")
        parser.add_argument('--x-label',
                            required=False,
                            help="The label on the x-axis. "
                            "Default is the name of the x-axis entry.")
        parser.add_argument('--y-label',
                            required=False,
                            help="The label on the y-axis. "
                            "Default is not having a label.")
        parser.add_argument('--linewidth',
                            required=False,
                            type=float,
                            help="The width of the lines plotted.",
                            default=plt.rcParams['lines.linewidth'])
        parser.add_argument(
            '--linestyles',
            required=False,
            nargs='+',
            help="The styles of the lines plotted. Must be one or "
            "one for each function.",
            choices=['solid', 'dashed', 'dashdot', 'dotted'],
            default=['solid'])
        parser.add_argument('--legend-ncols',
                            required=False,
                            type=int,
                            default=1,
                            help="Number of columns for the legend")
        parser.add_argument('--x-axis',
                            required=False,
                            help="If specified, make the range be the "
                            "values in this field of the subfile. Defaults to "
                            "the first column in the dat file.")
        parser.add_argument('--y-logscale',
                            required=False,
                            action='store_true',
                            help="If specified, set the y-axis to log scale.")

        parser.add_argument('--title',
                            required=False,
                            help="Title on the graph.")
        parser.add_argument('--fontsize',
                            required=False,
                            type=int,
                            default=16,
                            help="The font size on the plots")

        parser.add_argument(
            '--labels',
            required=False,
            type=json.loads,
            help="R|A string containing a JSON dictionary to map some of\n"
            "the functions to specified labels. An example of what\n"
            "to pass to the argument (including the single quotes)\n"
            "is:\n"
            "'{\"Error(DivergenceCleaningField)\":\"$L_2(\\\\Phi)$\",\n"
            "\"Error(RestMassDensity)\":\"$L_2(\\\\mathcal{E}(\\\\rho))$\"}'\n"
            "\n"
            "Note that the double slashes (\\\\) in LaTeX are needed to \n"
            "properly render the math.")

    args = parser.parse_args()

    # Do not do additional argument parsing if we are only printing file
    # contents or writing to a text file.
    if args.legend_only or args.write_dat:
        return args

    if args.functions is None:
        args.legend_only = True
        return args

    num_functions = len(args.functions)
    num_linestyles = len(args.linestyles)

    # Make sure for each specified label a function with that name exists
    if args.labels:
        functions = args.functions
        for key in args.labels:
            if not key in functions:
                parser.error(
                    "Unknown function '{}' specified with label '{}'. Known"
                    " functions are:\n{}".format(key, args.labels[key],
                                                 functions))

    # Make sure the correct number of line styles were supplied
    if num_linestyles != num_functions and num_linestyles != 1:
        parser.error(
            "You must supply either one line style or one line style per "
            "function. However, {} line styles with {} functions were parsed.\n"
            .format(num_linestyles, num_functions))

    # Convert line styles to what pyplot understands. There are difficulties
    # with parsing arguments like '--' and '-.' from the command line so
    # parsing strings and then converting is easier.
    if num_linestyles > 0:
        for linestyle in args.linestyles:
            linestyle = {
                'solid': '-',
                'dashed': '--',
                'dashdot': '-.',
                'dotted': '.'
            }[linestyle]

    return args


if __name__ == "__main__":
    open_and_process_file(**vars(parse_args()))
