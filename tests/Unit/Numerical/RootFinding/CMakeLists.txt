# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_ROOT_FINDER_TESTS
  Numerical/RootFinding/Test_OneDRootFinder.cpp)

set(SPECTRE_UNIT_NUMERICAL_TESTS "${SPECTRE_UNIT_NUMERICAL_TESTS};${SPECTRE_UNIT_ROOT_FINDER_TESTS}"
    PARENT_SCOPE)
