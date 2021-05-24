#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Extract appropriate compiler flags for compiling InfoAtLink.cpp

This script is invoked as if it was the compiler, compiling an object file from
the `InfoAtLink.cpp` source file. It extracts the flags and stores them in the
file `@CMAKE_BINARY_DIR@/Informer/InfoAtLink_flags.txt`. To compile
`InfoAtLink.cpp` at link time, read the flags from this file and pass them to
the compiler.
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__)
# The first positional argument is the compiler
parser.add_argument("compiler")
# Strip the object file name
parser.add_argument("-o", required=True)
# Strip stage selection for an object file
parser.add_argument("-c", action='store_true')
# Strip `-Werror` so `-fuse-ld=gold` compiler warnings from clang don't disrupt
# the linking. See issue: https://github.com/sxs-collaboration/spectre/issues/2703
parser.add_argument("-Werror", action='store_true')
# Parse the CLI args, discarding the arguments specified above and extracting
# the remaining compiler flags
args, compiler_flags = parser.parse_known_args()
# Discard the source file name
compiler_flags.pop()
# Write the extracted flags out to a file
output_filename = "@CMAKE_BINARY_DIR@/Informer/InfoAtLink_flags.txt"
with open(output_filename, "w") as output_file:
    output_file.write(" ".join(compiler_flags))
# Write a stub object file so CMake only re-runs this script when the
# build configuration changed.
open(args.o, 'w').close()
