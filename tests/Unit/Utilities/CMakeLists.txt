# Distributed under the MIT License.
# See LICENSE.txt for details.

set(UTILITIES_TESTS
    Utilities/BoostHelpers.cpp
    Utilities/ConstantExpressions.cpp
    Utilities/FakeVirtual.cpp
    Utilities/MakeArray.cpp
    Utilities/PrettyType.cpp
    Utilities/StdHelpers.cpp
    Utilities/TaggedTuple.cpp
    Utilities/TypeTraits.cpp)

set(SPECTRE_TESTS "${SPECTRE_TESTS};${UTILITIES_TESTS}" PARENT_SCOPE)

