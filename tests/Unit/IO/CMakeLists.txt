# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Importers)
add_subdirectory(Logging)

set(LIBRARY "Test_IO")

set(LIBRARY_SOURCES
  H5/Test_H5.cpp
  H5/Test_StellarCollapseEos.cpp
  H5/Test_VolumeData.cpp
  Observers/Test_ArrayComponentId.cpp
  Observers/Test_GetLockPointer.cpp
  Observers/Test_Initialize.cpp
  Observers/Test_RegisterElements.cpp
  Observers/Test_RegisterEvents.cpp
  Observers/Test_RegisterSingleton.cpp
  Observers/Test_Tags.cpp
  Observers/Test_ObservationId.cpp
  Observers/Test_ReductionObserver.cpp
  Observers/Test_TypeOfObservation.cpp
  Observers/Test_VolumeObserver.cpp
  Observers/Test_WriteSimpleData.cpp
  )

add_test_library(
  ${LIBRARY}
  "IO"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DataStructures
  DomainStructure
  EventsAndTriggers
  IO
  Parallel
  Utilities
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

spectre_add_python_bindings_test(
  "Unit.IO.H5.Python"
  "H5/Test_H5.py"
  "unit;IO;H5;python"
  PyH5)

spectre_add_python_bindings_test(
  "Unit.IO.H5.VolumeData.Python"
  "H5/Test_VolumeData.py"
  "unit;IO;H5;Python"
  PyH5
  )
