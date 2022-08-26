#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import re
import json


def extract_first_template_parameter(string):
    """
    Extract the first template parameter from a string

    Assume you have the string passed in is:
      A<1, B<2, 4>, 3>, C, D, E<7, 8, F<9>>
    We want to extract:
      A<1, B<2, 4>, 3>
    To do this we count the depth of the angle brackets and stop when we get
    to the first comma where we have zero angle brackets.
    """
    if not (',' in string):
        return string

    depth = 0
    for i in range(0, len(string)):
        if depth == 0 and string[i] == ',':
            break
        if string[i] == '<':
            depth = depth + 1
        if string[i] == '>':
            assert (depth > 1)
            depth = depth - 1
    assert (depth == 0)
    return string[:i]


def generic_replacements(text):
    """
    Apply generic naming replacements for Chares and entry methods.
    """
    for i in range(len(text)):
        if text[i].startswith(
                "ENTRY CHARE") and 'invoke_iterable_action<' in text[i]:
            matches = re.findall(
                ", std::integral_constant<unsigned long, [0-9]+>", text[i])
            text[i] = text[i].replace('invoke_iterable_action<', '')
            text[i] = text[i].replace(str(matches[-1]), '')
            text[i] = text[i].replace(str(matches[-2]), '')
            text[i] = text[i].replace('>()', '()')
        if text[i].startswith("ENTRY CHARE") and 'simple_action<' in text[
                i] and ">()" in text[i]:
            text[i] = text[i].replace("simple_action<",
                                      "").replace(">()", "()")
        elif text[i].startswith("ENTRY CHARE") and 'simple_action<' in text[i]:
            # The case where simple_action is receiving arguments
            text[i] = text[i].replace("simple_action<", "").replace(">(", "(")
            matches = re.findall("\"(.*)\(", text[i])
            # Replace tuple arg with nothing since the argument is irrelevant
            text[i] = re.sub(
                "\".*\(.*\)",
                "\"" + extract_first_template_parameter(matches[-1]) + "()",
                text[i])
        if text[i].startswith("ENTRY CHARE") and 'threaded_action<' in text[
                i] and ">()" in text[i]:
            text[i] = text[i].replace("threaded_action<",
                                      "").replace(">()", "()")
        elif text[i].startswith(
                "ENTRY CHARE") and 'threaded_action<' in text[i]:
            # The case where threaded_action is receiving arguments
            text[i] = text[i].replace("threaded_action<",
                                      "").replace(">(", "(")
            matches = re.findall("\"(.*)\(", text[i])
            # Replace tuple arg with nothing since the argument is irrelevant
            text[i] = re.sub(
                "\".*\(.*\)",
                "\"" + extract_first_template_parameter(matches[-1]) + "()",
                text[i])

        # Simplify general parallel components
        if text[i].startswith("CHARE") and 'AlgorithmGroup<' in text[i]:
            text[i] = text[i].replace("AlgorithmGroup<",
                                      "").replace(",int>", "")
        if text[i].startswith("CHARE") and 'AlgorithmNodegroup<' in text[i]:
            text[i] = text[i].replace("AlgorithmNodegroup<",
                                      "").replace(",int>", "")
        if text[i].startswith("CHARE") and 'Parallel::GlobalCache<' in text[i]:
            text[i] = re.sub("Parallel::GlobalCache<.*\"", "GlobalCache\"",
                             text[i])
        if text[i].startswith(
                "CHARE") and 'Parallel::MutableGlobalCache<' in text[i]:
            text[i] = re.sub("Parallel::MutableGlobalCache<.*\"",
                             "MutableGlobalCache\"", text[i])
        if text[i].startswith("CHARE") and 'Parallel::Main<' in text[i]:
            text[i] = re.sub("Parallel::Main<.*\"", "Main\"", text[i])
        if text[i].startswith(
                "CHARE") and 'Parallel::detail::AtSyncIndicator<' in text[i]:
            text[i] = re.sub("Parallel::detail::AtSyncIndicator<.*\"",
                             "AtSyncIndicator\"", text[i])

        # This will need to be updated when we rename to
        # evolution::DgElementArray
        if text[i].startswith(
                "CHARE") and 'AlgorithmArray<DgElementArray' in text[i]:
            text[i] = re.sub("\".*\"", "\"DgElementArray\"", text[i])
        if text[i].startswith(
                "CHARE"
        ) and 'AlgorithmArray<elliptic::DgElementArray' in text[i]:
            text[i] = re.sub("\".*\"", "\"elliptic::DgElementArray\"", text[i])

        # Simplify observers and interpolator
        if text[i].startswith("CHARE") and (
            ('intrp::Interpolator<' in text[i]) or
            ('observers::Observer<' in text[i]) or
            ('observers::ObserverWriter<' in text[i])):
            text[i] = re.sub("<.*\"", "\"", text[i])

    return text


def user_replacements(text, json_replacements):
    """
    Apply user replacements from a JSON file.

    First basic textual replacements are performed, followed by regular
    expression replacements.
    """
    # if 'RegexReplace' in json_replacements:
    #     raise ValueError("Regular expression replacement not yet added")
    if len(json_replacements) != 2:
        raise ValueError("Expected only 2 entry in the JSON root dictionary "
                         "but got " + str(len(json_replacements)))
    basic_replacements = json_replacements['BasicReplace']
    regex_replacements = json_replacements['RegexReplace']

    for i in range(len(text)):
        for replacement_entry in basic_replacements:
            if len(basic_replacements[replacement_entry]) == 0:
                raise ValueError("Expected at least one replacement for " +
                                 replacement_entry)

            if replacement_entry in text[i]:
                for replacement in basic_replacements[replacement_entry]:
                    if len(replacement) != 2:
                        raise ValueError(
                            "Basic replacement must be exactly length 2, "
                            "[to_replace, replace_with], but got " +
                            str(len(replacement)) + " (" + str(replacement) +
                            ") for entry " + replacement_entry)
                    text[i] = text[i].replace(replacement[0], replacement[1])

        for replacement_entry in regex_replacements:
            if len(regex_replacements[replacement_entry]) == 0:
                raise ValueError("Expected at least one replacement for " +
                                 replacement_entry)

            if replacement_entry in text[i]:
                for replacement in regex_replacements[replacement_entry]:
                    if len(replacement) != 2:
                        raise ValueError(
                            "Regex replacement must be exactly length 2, "
                            "[to_replace, replace_with], but got " +
                            str(len(replacement)) + " (" + str(replacement) +
                            ") for entry " + replacement_entry)
                    text[i] = re.sub(replacement[0], replacement[1], text[i])
    return text


def process_sts_file(filename, output, replacements_json_file):
    with open(filename, "r") as file:
        text = file.readlines()

    text = generic_replacements(text)
    if replacements_json_file:
        text = user_replacements(text,
                                 json.load(open(replacements_json_file, 'r')))

    with open(output, "w") as file:
        file.writelines(text)


def parse_args():
    """Parse command line arguments
    """
    import argparse as ap

    parser = ap.ArgumentParser(
        description="Process Charm++ Projections .sts (not .sum.sts) files to "
        "make the entry method and Chare names easier to read in the GUI. Long "
        "template names are not rendered fully making it impossible to figure "
        "out what Action and Chare was being measured. The standard entry "
        "methods like invoke_iterable_action and simple_action are "
        "simplified by default, but further textual and regular expression "
        "replacements can be specified in a JSON file.")
    parser.add_argument('filename',
                        help="The Charm++ Projections .sts file to read.")
    parser.add_argument(
        '--output',
        '-o',
        required=True,
        help="The output file to write to. Note that you will need to replace "
        "Charm++'s .sts file with the output file and the names must match.")
    parser.add_argument(
        '--replacements-json-file',
        required=False,
        help="A JSON file listing textual and regular expression replacements."
        " The file must specify \"BasicReplace\" and \"RegexReplace\" "
        "dictionaries. Each dictionary has keys that are the name of the "
        "replacement (unused in any searches). For BasicReplace the value is "
        "a list of two-element lists, the first entry in the nested "
        "two-element list is the string to replace and the second what to "
        "replace it with. An example entry is: \n"
        "'\"Actions::MutateApply\": [[\"Actions::MutateApply<\", \"\"], "
        "[\">()\", \"()\"]]'"
        " \nwhere if the line contains \"Actions::MutateApply<\" it and "
        "\">()\" are replaced. The regular expression is structured "
        "similarly but the entire regex match is replaced.")
    return parser.parse_args()


if __name__ == "__main__":
    process_sts_file(**vars(parse_args()))
