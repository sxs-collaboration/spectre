# Distributed under the MIT License.
# See LICENSE.txt for details.

# These paths should be relative to the input file directory passed to
# `add_input_file_tests`
set(INPUT_FILE_WHITELIST
    "PreprocessCceWorldtube/PreprocessCceWorldtube.yaml"
    "PreprocessCceWorldtube/AdmFirstOrderDriverPreprocessCceWorldtube.yaml"
    "PreprocessCceWorldtube/AdmSecondOrderDriverPreprocessCceWorldtube.yaml")

add_input_file_tests("${CMAKE_SOURCE_DIR}/tests/InputFiles/" "${INPUT_FILE_WHITELIST}")
