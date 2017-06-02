# Distributed under the MIT License.
# See LICENSE.txt for details.

set(UTILITIES_TESTS
    Utilities/BoostHelpers.cpp
    Utilities/ConstantExpressions.cpp
    Utilities/TypeTraits.cpp)

set(SPECTRE_TESTS "${SPECTRE_TESTS};${UTILITIES_TESTS}" PARENT_SCOPE)

