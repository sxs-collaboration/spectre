# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DataStructures)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ApplyMatrices.cpp
  CompressedMatrix.cpp
  CompressedVector.cpp
  DynamicBuffer.cpp
  DynamicMatrix.cpp
  DynamicVector.cpp
  FloatingPointType.cpp
  Index.cpp
  IndexIterator.cpp
  LeviCivitaIterator.cpp
  SliceIterator.cpp
  StripeIterator.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyMatrices.hpp
  BoostMultiArray.hpp
  CachedTempBuffer.hpp
  ComplexDataVector.hpp
  ComplexDiagonalModalOperator.hpp
  ComplexModalVector.hpp
  CompressedMatrix.hpp
  CompressedVector.hpp
  DataVector.hpp
  DiagonalModalOperator.hpp
  DynamicBuffer.hpp
  DynamicMatrix.hpp
  DynamicVector.hpp
  FixedHashMap.hpp
  FloatingPointType.hpp
  IdPair.hpp
  Index.hpp
  IndexIterator.hpp
  LeviCivitaIterator.hpp
  LinkedMessageId.hpp
  LinkedMessageQueue.hpp
  MathWrapper.hpp
  Matrix.hpp
  ModalVector.hpp
  SliceIterator.hpp
  SliceTensorToVariables.hpp
  SliceVariables.hpp
  SpinWeighted.hpp
  StaticDeque.hpp
  StaticMatrix.hpp
  StaticVector.hpp
  StripeIterator.hpp
  TaggedContainers.hpp
  Tags.hpp
  TempBuffer.hpp
  Transpose.hpp
  Variables.hpp
  VariablesTag.hpp
  VectorImpl.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blaze
  Boost::boost
  Charmxx::pup
  ErrorHandling
  Options
  Serialization
  Utilities
  )

add_subdirectory(Blaze)
add_subdirectory(DataBox)
add_subdirectory(Python)
add_subdirectory(Tags)
add_subdirectory(Tensor)
