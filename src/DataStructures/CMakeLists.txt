# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DataStructures)

set(LIBRARY_SOURCES
    DataVector.cpp
    Index.cpp
    IndexIterator.cpp
    LeviCivitaIterator.cpp
    SliceIterator.cpp
    StripeIterator.cpp
    Tensor/TensorData.cpp
    VariablesHelpers.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE Domain
  INTERFACE ErrorHandling
  )
