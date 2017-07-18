# Distributed under the MIT License.
# See LICENSE.txt for details.

set(UTILITIES_TESTS
    Utilities/Test_Blas.cpp
    Utilities/Test_BoostHelpers.cpp
    Utilities/Test_ConstantExpressions.cpp
    Utilities/Test_Deferred.cpp
    Utilities/Test_FakeVirtual.cpp
    Utilities/Test_FileSystem.cpp
    Utilities/Test_MakeArray.cpp
    Utilities/Test_Math.cpp
    Utilities/Test_PrettyType.cpp
    Utilities/Test_StdHelpers.cpp
    Utilities/Test_TaggedTuple.cpp
    Utilities/Test_TypeTraits.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${UTILITIES_TESTS}" PARENT_SCOPE)

