# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ArraySize.hpp
  CanBeCopyConstructed.hpp
  CreateGetStaticMemberVariableOrDefault.hpp
  CreateGetTypeAliasOrDefault.hpp
  CreateHasStaticMemberVariable.hpp
  CreateHasTypeAlias.hpp
  CreateIsCallable.hpp
  FastPointerCast.hpp
  FunctionInfo.hpp
  GetFundamentalType.hpp
  HasEquivalence.hpp
  HasInequivalence.hpp
  IsA.hpp
  IsCallable.hpp
  IsComplexOfFundamental.hpp
  IsInteger.hpp
  IsIterable.hpp
  IsMaplike.hpp
  IsStdArray.hpp
  IsStdArrayOfSize.hpp
  IsStreamable.hpp
  RemoveReferenceWrapper.hpp
  TypeTraits.hpp
  )
