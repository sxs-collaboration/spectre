# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DataStructures)

set(LIBRARY_SOURCES
  Index.cpp
  IndexIterator.cpp
  LeviCivitaIterator.cpp
  SliceIterator.cpp
  StripeIterator.cpp
  Tensor/EagerMath/OrthonormalOneform.cpp
  Tensor/TensorData.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blaze
  Boost::boost
  ErrorHandling
  )

add_subdirectory(Python)
