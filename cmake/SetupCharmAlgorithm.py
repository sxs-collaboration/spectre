#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os.path
import sys


def create_interface_file(args):
    """
    Generate the Charm++ interface file for the algorithm
    """
    ci_str = "// Distributed under the MIT License.\n" \
             "// See LICENSE.txt for details.\n\n" \
             "module Algorithm%s {\n" \
             "  include \"Utilities/TaggedTuple.hpp\";\n" \
             "  include \"Parallel/ConstGlobalCache.decl.h\";\n" \
             "\n" \
             "  template <typename ParallelComponent,\n" \
             "            typename SpectreArrayIndex>\n" \
             % args['algorithm_name']
    # The chare type needs to be checked
    if args['algorithm_type'] == "array":
        ci_str += "  array [SpectreArrayIndex]"
    else:
        ci_str += "  %s [migratable]" % (args['algorithm_type'])

    ci_str += " Algorithm%s {\n" \
              "    entry Algorithm%s(" \
              "Parallel::CProxy_ConstGlobalCache<\n" \
              "                      METAVARIABLES_FROM_COMPONENT>);\n" \
              "\n" \
              "    template <typename Action, typenameLDOTLDOTLDOT Args>\n" \
              "    entry void simple_action(\n" \
              "               std::tuple<COMPUTE_VARIADIC_ARGS>& args);\n" \
              "\n" \
              "    template <typename Action>\n" \
              "    entry void simple_action();\n" \
              "\n" \
              "    template <typename Action, typename Arg>\n" \
              "    entry [reductiontarget] void reduction_action(Arg arg);\n" \
              "\n" \
              "    entry void perform_algorithm();\n" \
              "\n" % (args['algorithm_name'], args['algorithm_name'])

    if (args['algorithm_type'] == "nodegroup"):
        ci_str += \
            "    template <typename Action, typenameLDOTLDOTLDOT Args>\n" \
            "    entry void threaded_action(\n" \
            "               std::tuple<COMPUTE_VARIADIC_ARGS>& args);\n" \
            "\n" \
            "    template <typename Action>\n" \
            "    entry void threaded_action();\n" \
            "\n"

    # A bug in Charm++ prevents entry methods with default argument values.
    # The workaround is to have two entry methods and default the argument in
    # the AlgorithmImpl member function.
    if (args['algorithm_type'] == "group"
            or args['algorithm_type'] == "nodegroup"):
        ci_str += \
            "    template <typename ReceiveTag, typename ReceiveData_t>\n" \
            "    entry void receive_data(ReceiveTag_temporal_id&,\n" \
            "                            ReceiveData_t&,\n" \
            "                            bool enable_if_disabled);\n"
        ci_str += \
            "    template <typename ReceiveTag, typename ReceiveData_t>\n" \
            "    entry void receive_data(ReceiveTag_temporal_id&,\n" \
            "                            ReceiveData_t&);\n"
    else:
        ci_str += \
            "    template <typename ReceiveTag, typename ReceiveData_t>\n" \
            "    entry void receive_data(\n" \
            "                            ReceiveTag_temporal_id&,\n" \
            "                            ReceiveData_t&,\n" \
            "                            bool enable_if_disabled = false);\n"

    ci_str += "\n" "    entry void set_terminate(bool);\n" "  }\n" "}\n"

    file_name = "%s/Algorithm%s.ci" % (args['build_dir'],
                                       args['algorithm_name'])
    if os.path.isfile(file_name) and open(file_name, "r").read() == ci_str:
        return 255
    open(file_name, "w").write(ci_str)
    return 0


def create_header_file(args):
    """
    Generate the C++ header file for the algorithm that wraps the
    Charm++ generated decl.h and def.h files
    """
    # Write include files and decl.h
    header_str = "// Distributed under the MIT License.\n" \
                 "// See LICENSE.txt for details.\n" \
                 "\n" \
                 "#ifdef GCC\n" \
                 "#pragma GCC system_header\n" \
                 "#endif\n" \
                 "\n#pragma once\n" \
                 "\n" \
                 "#include \"Parallel/Algorithm.hpp\"\n" \
                 "#include \"Parallel/ArrayIndex.hpp\"\n\n" \
                 "#include \"Algorithms/Algorithm%s.decl.h\"\n\n" % \
                 args['algorithm_name']
    # Write "ChareType" struct
    header_str += \
        "namespace Parallel {\n" \
        "namespace Algorithms {\n" \
        "struct %s {\n" \
        "  template <typename ParallelComponent,\n" \
        "            typename SpectreArrayIndex>\n" \
        "  using cproxy = CProxy_Algorithm%s<ParallelComponent,\n" \
        "                           SpectreArrayIndex>;\n" \
        "\n" \
        "  template <typename ParallelComponent,\n" \
        "            typename SpectreArrayIndex>\n" \
        "  using cbase = CBase_Algorithm%s<ParallelComponent,\n" \
        "                          SpectreArrayIndex>;\n" \
        "\n" \
        "  template <typename ParallelComponent,\n" \
        "            typename SpectreArrayIndex>\n" \
        "  using algorithm_type = Algorithm%s<ParallelComponent,\n" \
        "                             SpectreArrayIndex>;\n" \
        "\n" \
        "  template <typename ParallelComponent,\n" \
        "            typename SpectreArrayIndex>\n" \
        "  using ckindex = CkIndex_Algorithm%s<ParallelComponent,\n" \
        "                             SpectreArrayIndex>;\n" \
        "};\n" \
        "}  // namespace Algorithms\n" \
        "}  // namespace Parallel\n\n" % \
        (args['algorithm_name'], args['algorithm_name'],
         args['algorithm_name'], args['algorithm_name'],
         args['algorithm_name'])
    # Write Algorithm class
    header_str += \
        "template <typename ParallelComponent,\n" \
        "          typename SpectreArrayIndex>\n" \
        "class Algorithm%s\n" \
        "    : public CBase_Algorithm%s<ParallelComponent, \n" \
        "                      SpectreArrayIndex>,\n" \
        "      public Parallel::AlgorithmImpl<ParallelComponent,\n" \
        "                       typename ParallelComponent::action_list> {\n" \
        "  using algorithm = Parallel::Algorithms::%s;\n" \
        " public:\n" \
        "  using Parallel::AlgorithmImpl<ParallelComponent,\n" \
        "                  typename ParallelComponent::action_list\n" \
        "                  >::AlgorithmImpl;\n" \
        "};\n\n" % (args['algorithm_name'],
                    args['algorithm_name'], args['algorithm_name'])
    # Write include of the def file, but including only the template definitions
    header_str += "#define CK_TEMPLATES_ONLY\n" \
                  "#include \"Algorithms/Algorithm%s.def.h\"\n" \
                  "#undef CK_TEMPLATES_ONLY\n" % args['algorithm_name']

    file_name = "%s/Algorithm%s.hpp" % (args['build_dir'],
                                        args['algorithm_name'])
    if os.path.isfile(file_name) and open(file_name, "r").read() == header_str:
        return 255
    open(file_name, "w").write(header_str)
    return 0


def parse_args():
    """
    Parse the command line arguments
    """
    import argparse as ap
    parser = ap.ArgumentParser(
        description='Generate Charm++ ci file and header file for an Algorithm',
        formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algorithm_name', required=True, help="The name of the algorithm")
    parser.add_argument(
        '--algorithm_type',
        required=True,
        choices=['chare', 'array', 'group', 'nodegroup'],
        help="The type of algorithm to generate")
    parser.add_argument(
        '--build_dir',
        required=True,
        help="Absolute path to the build directory")
    return vars(parser.parse_args())


if __name__ == "__main__":
    input_args = parse_args()
    ci_existed = create_interface_file(input_args) == 255
    header_existed = create_header_file(input_args) == 255
    if ci_existed and header_existed:
        sys.exit(255)
