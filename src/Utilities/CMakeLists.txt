# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Utilities)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Blas.cpp
  FileSystem.cpp
  Formaline.cpp
  OptimizerHacks.cpp
  PrettyType.cpp
  Rational.cpp
  WrapText.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Algorithm.hpp
  Array.hpp
  Blas.hpp
  Blaze.hpp
  BoostHelpers.hpp
  CachedFunction.hpp
  CloneUniquePtrs.hpp
  ConstantExpressions.hpp
  ContainerHelpers.hpp
  DereferenceWrapper.hpp
  EqualWithinRoundoff.hpp
  FakeVirtual.hpp
  FileSystem.hpp
  ForceInline.hpp
  Formaline.hpp
  FractionUtilities.hpp
  Functional.hpp
  GenerateInstantiations.hpp
  GetOutput.hpp
  Gsl.hpp
  Literals.hpp
  MakeArray.hpp
  MakeSignalingNan.hpp
  MakeString.hpp
  MakeVector.hpp
  MakeWithValue.hpp
  Math.hpp
  NoSuchType.hpp
  Numeric.hpp
  OptimizerHacks.hpp
  Overloader.hpp
  PointerVector.hpp
  PrettyType.hpp
  PrintHelpers.hpp
  ProtocolHelpers.hpp
  Rational.hpp
  Registration.hpp
  Requires.hpp
  Spherepack.hpp
  StaticCache.hpp
  StdArrayHelpers.hpp
  StdHelpers.hpp
  StlStreamDeclarations.hpp
  TMPL.hpp
  TaggedTuple.hpp
  TmplDebugging.hpp
  TmplDigraph.hpp
  Tuple.hpp
  TupleSlice.hpp
  TypeTraits.hpp
  VectorAlgebra.hpp
  WrapText.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  Blaze
  Boost::boost
  Brigand
  ErrorHandling
  Libxsmm
  )

add_subdirectory(TypeTraits)

if (NOT USE_FORMALINE)
  set_source_files_properties(
    Formaline.cpp
    PROPERTIES
    COMPILE_DEFINITIONS SPECTRE_NO_FORMALINE)
endif (NOT USE_FORMALINE)
