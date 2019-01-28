#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import re

# A list of all the allowed ctest labels/Catch tags for tests
allowed_tags = [
                "Actions",
                "ApparentHorizons",
                "Burgers",
                "CompilationTest",
                "ControlSystem",
                "DataStructures",
                "Domain",
                "Elasticity",
                "Elliptic",
                "EquationsOfState",
                "ErrorHandling",
                "Evolution",
                "Fluxes",
                "GeneralizedHarmonic",
                "GrMhd",
                "H5",
                "Hydro",
                "IO",
                "Informer",
                "LinearAlgebra",
                "LinearOperators",
                "LinearSolver",
                "M1Grey",
                "NumericalAlgorithms",
                "Observers",
                "Options",
                "Parallel",
                "PointwiseFunctions",
                "Pypp",
                "RelativisticEuler",
                "RootFinding",
                "Serialization",
                "SlopeLimiters",
                "Spectral",
                "Time",
                "Unit",
                "Utilities",
                "VariableFixing",
               ]

# Words disallowed in tests
disallowed_test_name_portions = ["Functors"]

# Allowed test attributes
allowed_test_attributes = ["TimeOut",
                           "OutputRegex"]

# All the timeout times for the different types of tests. The order here
# matters. Whichever time is specified last is what will be used for the
# test if it is of multiple types.
default_tag_timeouts = [("unit", 5)]

allowed_tags = [x.lower() for x in allowed_tags]
allowed_test_attributes = [x.lower() for x in allowed_test_attributes]

def parse_source_file(file_name):
    file_string = open(file_name, "r").read()
    test_regex = re.compile("(\/\/ \[\[.*?)?SPECTRE_TEST_CASE\((.*?)\) {",
                            re.DOTALL)
    for (attributes, test_name) in re.findall(test_regex, file_string):
        # Capture the name of the test into the first group and the tags into
        # the second. For example,
        # "Unit.My.Test", "[Unit][My]"
        # group(1) == Unit.My.Test
        # group(2) == [Unit][My]
        parsed_name = re.search("\"(.*)\",[\s]*\"(.*)\"", test_name)
        test_name = parsed_name.group(1)
        for disallowed_name in disallowed_test_name_portions:
            if test_name.lower().find(disallowed_name.lower()) != -1:
                print("\nERROR: Found disallowed portion of a test name '%s' "
                      "the test named '%s' in the file %s." %
                      (disallowed_name, test_name, file_name))
                exit(1)
        test_tags = parsed_name.group(2).lower().replace("[", "")[:-1]
        test_tags = test_tags.split("]")
        for test_tag in test_tags:
            if not test_tag in allowed_tags:
                print("\nERROR: The tag '%s' is not allowed but was found in "
                      "the test '%s' in the file '%s'. To allow it add it to "
                      "the 'allowed_tags' list in "
                      "$SPECTRE_ROOT/cmake/SpectreParseTests.py. The currently "
                      "allowed tags are:\n%s\n\n" %
                      (test_tag, test_name, file_name, allowed_tags))
                exit(1)
        test_timeout = -1
        for (tag, timeout) in default_tag_timeouts:
            if tag in test_tags:
                test_timeout = timeout

        # Parse the test attributes
        output_regex = ''

        all_attributes_by_name = re.findall("\[\[([^,]+), (.*?)\]\]",
                                            attributes, re.DOTALL)
        for attribute in all_attributes_by_name:
            if not attribute[0].lower() in allowed_test_attributes:
                print("\nERROR: Found unknown test attribute '%s' applied "
                      "to test '%s' in file %s." %
                      (attribute[0], test_name, file_name))
                exit(1)

            if attribute[0].lower() == "timeout":
                test_timeout = attribute[1]

            if attribute[0].lower() == "outputregex":
                output_regex = attribute[1].replace("\n//", "")

        open("%s.timeout" % test_name, "w").write("%s" % test_timeout)
        open("%s.output_regex" % test_name, "w").write("%s" % output_regex)


if __name__ == '__main__':
    import sys
    source_files = sys.argv[1:]
    for file in source_files:
        parse_source_file(file)
