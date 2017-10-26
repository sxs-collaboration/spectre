# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DataStructures)

set(LIBRARY_SOURCES
    DataVector.cpp
    Index.cpp
    IndexIterator.cpp
    Matrix.cpp
    SliceIterator.cpp
    StripeIterator.cpp
    VariablesHelpers.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE ErrorHandling
  )
