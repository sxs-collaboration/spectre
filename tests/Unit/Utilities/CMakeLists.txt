# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Utilities")

set(LIBRARY_SOURCES
  Test_Algorithm.cpp
  Test_Array.cpp
  Test_Blas.cpp
  Test_CachedFunction.cpp
  Test_CallWithDynamicType.cpp
  Test_CartesianProduct.cpp
  Test_CleanupRoutine.cpp
  Test_CloneUniquePtrs.cpp
  Test_ConstantExpressions.cpp
  Test_ContainerHelpers.cpp
  Test_DereferenceWrapper.cpp
  Test_EqualWithinRoundoff.cpp
  Test_FileSystem.cpp
  Test_Formaline.cpp
  Test_FractionUtilities.cpp
  Test_Functional.cpp
  Test_GetOutput.cpp
  Test_Gsl.cpp
  Test_MakeArray.cpp
  Test_MakeSignalingNan.cpp
  Test_MakeString.cpp
  Test_MakeVector.cpp
  Test_MakeWithValue.cpp
  Test_Math.cpp
  Test_Numeric.cpp
  Test_OptionalHelpers.cpp
  Test_Overloader.cpp
  Test_ParallelInfo.cpp
  Test_PrettyType.cpp
  Test_ProtocolHelpers.cpp
  Test_Rational.cpp
  Test_Registration.cpp
  Test_Requires.cpp
  Test_StaticCache.cpp
  Test_StdArrayHelpers.cpp
  Test_StdHelpers.cpp
  Test_StlBoilerplate.cpp
  Test_TaggedTuple.cpp
  Test_TMPL.cpp
  Test_TMPLDocumentation.cpp
  Test_Tuple.cpp
  Test_TupleSlice.cpp
  Test_VectorAlgebra.cpp
  Test_WrapText.cpp
  )

add_subdirectory(TypeTraits)
add_subdirectory(Serialization)

add_test_library(
  ${LIBRARY}
  "Utilities"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  CoordinateMaps
  DataStructures
  Options
  Serialization
  Utilities
)
