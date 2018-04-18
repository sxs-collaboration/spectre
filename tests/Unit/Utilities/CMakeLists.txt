# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Utilities")

set(LIBRARY_SOURCES
  Test_Array.cpp
  Test_Blas.cpp
  Test_BoostHelpers.cpp
  Test_CachedFunction.cpp
  Test_ConstantExpressions.cpp
  Test_ContainerHelpers.cpp
  Test_DereferenceWrapper.cpp
  Test_EqualWithinRoundoff.cpp
  Test_FakeVirtual.cpp
  Test_FileSystem.cpp
  Test_FractionUtilities.cpp
  Test_Gsl.cpp
  Test_MakeArray.cpp
  Test_MakeVector.cpp
  Test_MakeWithValue.cpp
  Test_Math.cpp
  Test_Overloader.cpp
  Test_PrettyType.cpp
  Test_Rational.cpp
  Test_Requires.cpp
  Test_StdArrayHelpers.cpp
  Test_StdHelpers.cpp
  Test_TaggedTuple.cpp
  Test_TMPL.cpp
  Test_Tuple.cpp
  Test_TypeTraits.cpp
  )

add_test_library(
  ${LIBRARY}
  "Utilities"
  "${LIBRARY_SOURCES}"
  "Utilities"
  )
