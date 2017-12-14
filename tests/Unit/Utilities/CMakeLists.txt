# Distributed under the MIT License.
# See LICENSE.txt for details.

set(UTILITIES_TESTS
    Utilities/Test_Blas.cpp
    Utilities/Test_BoostHelpers.cpp
    Utilities/Test_CachedFunction.cpp
    Utilities/Test_ConstantExpressions.cpp
    Utilities/Test_ContainerHelpers.cpp
    Utilities/Test_Deferred.cpp
    Utilities/Test_DereferenceWrapper.cpp
    Utilities/Test_EqualWithinRoundoff.cpp
    Utilities/Test_FakeVirtual.cpp
    Utilities/Test_FileSystem.cpp
    Utilities/Test_FractionUtilities.cpp
    Utilities/Test_Gsl.cpp
    Utilities/Test_MakeArray.cpp
    Utilities/Test_MakeVector.cpp
    Utilities/Test_Math.cpp
    Utilities/Test_Overloader.cpp
    Utilities/Test_PrettyType.cpp
    Utilities/Test_Requires.cpp
    Utilities/Test_StdHelpers.cpp
    Utilities/Test_TaggedTuple.cpp
    Utilities/Test_TMPL.cpp
    Utilities/Test_Tuple.cpp
    Utilities/Test_TypeTraits.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${UTILITIES_TESTS}" PARENT_SCOPE)
