#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import subprocess
import os
import sys
import re

# List of all the clang-format versions we are willing to use
# The general case is needed on a Mac
clang_format_list = ["git-clang-format",
                     "git-clang-format-6.0",
                     "git-clang-format-4.0",
                     "git-clang-format-3.9",
                     "git-clang-format-3.8"]

git_executable = "@GIT_EXECUTABLE@"


def which(program):
    """ used to find the binary; if it does not exist returns None """
    def is_bin(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_bin(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            bin_file = os.path.join(path, program)
            if is_bin(bin_file):
                return bin_file

    return None

# Check each of the allowed versions
clang_format = ""
for version in clang_format_list:
    clang_format = which(version)
    if clang_format:
        break

if not clang_format:
    sys.exit(0)

# extract the clang-format-a.b part of the string
clang_format = re.search(".*\/git-(clang-format[-0-9\.]*)",
                         clang_format).group(1)

# Check to make sure version of clang-format we are using is sufficiently new
clang_format_version = subprocess.check_output([clang_format, "--version"])
clang_format_version = re.search("clang-format version ([0-9]+)\.([0-9]+)",
                                 str(clang_format_version))

if (int(clang_format_version.group(1)) < 4 and
        int(clang_format_version.group(2)) < 8):
    print("clang-format version %s.%s is too low. Must have at least 3.8" %
              (clang_format_version.group(1), clang_format_version.group(2)))
    sys.exit(0)

output = subprocess.check_output([git_executable, clang_format, "--diff"
                                  ]).decode('ascii')

if output not in ['\n', '', 'no modified files to format\n',
                  'clang-format did not modify any files\n']:
    output_file_name = "@CMAKE_SOURCE_DIR@/.clang_format_diff.patch"
    output_file = open(output_file_name, 'w')
    output_file.write("%s" % (output))
    output_file.close()
    print("\nWARNING:\n"
          "ClangFormat found differences. The diff output has been written to"
          "\n'%s'\n"
          "You can apply the patch as-is using the following two commands:\n"
          "  cd @CMAKE_SOURCE_DIR@\n"
          "  git apply %s\n"
          "and then staging the modified files and amending your original "
          "commit.\n" %
          (output_file_name, output_file_name))
    sys.exit(1)

sys.exit(0)
