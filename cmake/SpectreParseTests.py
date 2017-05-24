#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import re

# All the timeout times for the different types of tests. The order here
# matters. Whichever time is specified last is what will be used for the
# test if it is of multiple types.
default_tag_timeouts = [("unit", 2)]


def parse_source_file(file_name):
    file_string = open(file_name, "r").read()
    test_regex = re.compile("(\/\/ \[\[.*?)?TEST_CASE\((.*?)\) {", re.DOTALL)
    for (attributes, test_name) in re.findall(test_regex, file_string):
        # Capture the name of the test into the first group and the tags into
        # the second. For example,
        # "Unit.My.Test", "[Unit][My]"
        # group(1) == Unit.My.Test
        # group(2) == [Unit][My]
        parsed_name = re.search("\"(.*)\",[\s]*\"(.*)\"", test_name)
        test_name = parsed_name.group(1)
        test_tags = parsed_name.group(2).lower().replace("[", "")[:-1]
        test_tags = test_tags.split("]")
        test_timeout = -1
        for (tag, timeout) in default_tag_timeouts:
            if tag in test_tags:
                test_timeout = timeout
        # parse attributes for time
        explicit_timeout = re.search("\[\[TimeOut, ([0-9]+)\]\]", attributes)
        if explicit_timeout:
            test_timeout = explicit_timeout.group(1)
        output_regex = re.search("\[\[OutputRegex, (.*?)\]\]",
                                attributes, re.DOTALL)
        if output_regex:
            output_regex = output_regex.group(1).replace("\n//", "")

        open("%s.timeout" % test_name, "w").write("%s" % test_timeout)
        open("%s.output_regex" % test_name, "w").write("%s" % output_regex)


if __name__ == '__main__':
    import sys
    source_files = sys.argv[1:]
    for file in source_files:
        parse_source_file(file)
