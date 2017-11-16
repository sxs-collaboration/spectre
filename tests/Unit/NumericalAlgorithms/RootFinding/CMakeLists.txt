# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_ROOT_FINDER
    NumericalAlgorithms/RootFinding/Test_OneDRootFinder.cpp
    NumericalAlgorithms/RootFinding/Test_QuadraticEquation.cpp
   )

 set(SPECTRE_UNIT_NUMERICAL_ALGORITHMS
   "${SPECTRE_UNIT_NUMERICAL_ALGORITHMS};${SPECTRE_UNIT_ROOT_FINDER}"
    PARENT_SCOPE)
