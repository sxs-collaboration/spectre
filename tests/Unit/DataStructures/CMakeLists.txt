# Distributed under the MIT License.
# See LICENSE.txt for details.

set(DATA_STRUCTURES_TESTS
    DataStructures/DataVector.cpp
    DataStructures/Index.cpp
    DataStructures/IndexIterator.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${DATA_STRUCTURES_TESTS}" PARENT_SCOPE)
