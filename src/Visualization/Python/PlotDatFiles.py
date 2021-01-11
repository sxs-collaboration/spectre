#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import textwrap

# Set back to normal LaTeX math font...
import matplotlib.style
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


def read_and_plot_data(args, dat_file):
    """
    Read the data from the dat file and generate a plot.

    Checks that various preconditions are met and throws
    an error if they are not.
    """
    legend = [x.decode() for x in dat_file.attrs['Legend']]
    all_data = np.asarray(dat_file)
    if args['x_axis'] == None:
        args['x_axis'] = legend[0]
    elif not args['x_axis'] in legend:
        sys.exit("Unknown x-axis function '%s'\nKnown functions are:\n%s" %
                 (args['x_axis'], str(legend)))

    # make sure all requested functions are in the file:
    for i_function in range(len(args['functions'])):
        function = args['functions'][i_function]
        if not function in legend:
            sys.exit("Unknown function '%s'\nKnown functions are:\n%s" %
                     (function, str(legend)))

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
    plt.savefig(args['o'] + '.pdf',
                transparent=True,
                format='pdf',
                bbox_inches='tight')
    return None


def process_subfile(args, dat_file):
    """
    Given the read in subfile.

    If --legend-only was specified then the legend is printed
    and execution terminated.

    If --write-dat is specified then the data is written to a text dat file.

    Generates a plot and writes it to disk using read_and_plot_data.
    """
    legend = [x.decode() for x in dat_file.attrs['Legend']]
    if args['legend_only']:
        print("The legend in the dat subfile is:\n%s" % legend)
        return None

    if not (args['write_dat'] is None):
        print("Writing to text dat file '%s'" % args['write_dat'])
        np.savetxt(args['write_dat'],
                   np.asarray(dat_file),
                   delimiter=' ',
                   header='\n'.join(
                       '{}' for _ in range(len(legend))).format(*legend))
        return None

    read_and_plot_data(args, dat_file)
    return None


def open_and_process_file(args):
    """
    Opens the specified HDF5 file.

    If no subfile is specified this function prints all available Dat
    subfiles in the root of the HDF5 and then exists. If a subfile is
    specified, the function forwards on to process_subfile()
    """
    def available_subfiles(h5file):
        available_files = ""
        for member in h5file.keys():
            if member.endswith(".dat"):
                available_files += "  " + str(member)
        return available_files

    with h5py.File(args['file']) as h5file:
        if args['subfile'] != None:
            dat_file = h5file.get(args['subfile'] + '.dat')
            if dat_file is None:
                raise Exception(
                    "Unable to open dat file '%s'. Available files are:\n%s" %
                    (args['subfile'] + '.dat', available_subfiles(h5file)))
            process_subfile(args, dat_file)
        else:
            print("Available dat files are:", available_subfiles(h5file))
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
    parser.add_argument('--file',
                        required=True,
                        help="The file to write",
                        type=str)
    parser.add_argument(
        '--subfile',
        required=False,
        help="The dat subfile to read excluding the .dat extension. "
        "If excluded all available dat subfiles will be printed.",
        type=str)
    parser.add_argument('--legend-only',
                        required=False,
                        action='store_true',
                        help="If specified, only print out the legend")
    parser.add_argument(
        '--write-dat',
        required=False,
        help="If specified, writes the entire dat file to a text Dat file with "
        "spaces as the delimiter and exits.",
        type=str)
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
                        type=str,
                        help="The label on the x-axis. "
                        "Default is the name of the x-axis entry.")
    parser.add_argument('--linewidth',
                        required=False,
                        type=float,
                        help="The width of the lines plotted.",
                        default=plt.rcParams['lines.linewidth'])
    parser.add_argument('--linestyles',
                        required=False,
                        nargs='+',
                        type=str,
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
                        type=str,
                        help="If specified, make the range be the "
                        "values is this field of the subfile. Defaults to "
                        "the first column in the dat file.")
    parser.add_argument('--y-logscale',
                        required=False,
                        action='store_true',
                        help="If specified, set the y-axis to log scale.")

    parser.add_argument('--title',
                        required=False,
                        type=str,
                        help="Title on the graph.")
    parser.add_argument('--fontsize',
                        required=False,
                        type=int,
                        default=16,
                        help="The font size on the plots")

    parser.add_argument('--functions',
                        required=False,
                        nargs='+',
                        help="The quantities to plot as a function "
                        "of '--x-axis'.")
    parser.add_argument(
        '--labels',
        required=False,
        type=json.loads,
        help="R|A string containing a JSON dictionary to map some of\n"
        "the functions to specified labels. An example of what\n"
        "to pass to the argument (including the single quotes)\n"
        "is:\n"
        "'{\"Error(DivergenceCleaningField)\":\"$L_2(\\\\Phi)$\",\n"
        "\"Error(RestMassDensity)\":\"$L_2(\\\\mathcal{E}(\\\\rho))$\"}'")

    parser.add_argument(
        '-o',
        required=False,
        type=str,
        default='plot',
        help="Name of the output file. The '.pdf' extension will "
        "be added automatically.")

    args = vars(parser.parse_args())

    # Do not do additional argument parsing if we are only printing file
    # contents or writing to a text file.
    if args['legend_only'] or args['write_dat']:
        return args

    if args['functions'] is None:
        args['legend_only'] = True
        return args

    num_functions = len(args['functions'])
    num_linestyles = len(args['linestyles'])

    # Make sure for each specified label a function with that name exists
    if args['labels']:
        functions = args['functions']
        for key in args['labels']:
            if not key in functions:
                print("\n".join(
                    textwrap.wrap(
                        "Unknown function '%s' specified with label '%s'. Known"
                        " functions are:\n%s" %
                        (key, args['labels'][key], functions),
                        width=80)) + '\n')
                parser.print_help()
                sys.exit(1)

    # Make sure the correct number of line styles were supplied
    if num_linestyles != num_functions and num_linestyles != 1:
        print("\n".join(
            textwrap.wrap(
                "You must supply either zero, one, or one line style per "
                "function but parsed %d line styles with %d functions." %
                (num_linestyles, num_functions),
                width=80)) + '\n')
        parser.print_help()
        sys.exit(1)

    # Convert line styles to what pyplot understands. There are difficulties
    # with parsing arguments like '--' and '-.' from the command line so
    # parsing strings and then converting is easier.
    if num_linestyles > 0:
        for linestyle in args['linestyles']:
            linestyle = {
                'solid': '-',
                'dashed': '--',
                'dashdot': '-.',
                'dotted': '.'
            }[linestyle]

    return args


if __name__ == "__main__":
    open_and_process_file(parse_args())
