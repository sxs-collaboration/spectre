# Distributed under the MIT License.
# See LICENSE.txt for details.

set(PARALLEL_TESTS
    Parallel/Test_Parallel.cpp
    Parallel/Test_PupStlCpp11.cpp
    )
set(SPECTRE_TESTS "${SPECTRE_TESTS};${PARALLEL_TESTS}" PARENT_SCOPE)

