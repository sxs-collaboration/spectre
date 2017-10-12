#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import re


def parse_file(input_file_name, output_file_name):
    """
    Parses a LaTeX file and replaces most tex commands with ones
    compatible with Doxygen and MathJAX
    """
    tex_string = open(input_file_name, 'r').read()
    # Replace equations, etc.
    out_string = re.sub("\\\\begin{(align\*?|equation\*?|eqnarray\*?)}",
                        r"\\f{\1}{", tex_string)
    out_string = re.sub("\\\\end{(align\*?|equation\*?|eqnarray\*?)}",
                        r"\\f}", out_string)
    # Replace section and subsection
    out_string = re.sub("\\\\section{([^}]+)}",
                        r"## \1", out_string)
    out_string = re.sub("\\\\subsection{([^}]+)}",
                        r"### \1", out_string)
    # Get inline math to work
    out_string = re.sub("([^\\\\])\$", r"\1\\f$", out_string)
    # Wrap \ref in math to get references to work
    out_string = re.sub("(\\\\(eq)?ref{[^}]+})", r"\\f$\1\\f$", out_string)
    # Remove all comment lines
    out_string = re.sub("([^\\\\])%.*", r"\1\n", out_string)
    # Get newcommand from tex file and add them to the output
    command_string = ""
    for newcommand in re.findall("(\\\\newcommand.*)\n", tex_string):
        command_string = "%s\\f$%s\\f$\n" % (command_string, newcommand)

    # Extract contents inside document
    is_doc = re.search(re.compile("\\\\begin{document}(.*)\\\\end{document}",
                                  re.DOTALL),
                       out_string)
    if is_doc:
        out_string = is_doc.group(1)

    out_string = "%s\n%s" % (command_string, out_string)
    open(output_file_name, 'w').write(out_string)


def parse_args():
    """
    Parse the command line arguments
    """
    import argparse as ap
    parser = ap.ArgumentParser(
        description='Do a simple conversion from LaTeX to Doxygen comment '
        'using MathJAX. The result will likely need some additional '
        'tweaking but the goal of this script is to automate the majority'
        ' of the work.',
        formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--tex-file', required=True, help="The name LaTeX file to process")
    parser.add_argument(
        '--output-file',
        required=True,
        help="The name of the file to write the output to")
    return vars(parser.parse_args())


if __name__ == "__main__":
    input_args = parse_args()
    parse_file(input_args['tex_file'], input_args['output_file'])
