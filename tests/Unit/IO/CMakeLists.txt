# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_IO")

set(LIBRARY_SOURCES
  Observers/Test_ArrayComponentId.cpp
  Observers/Test_Initialize.cpp
  Observers/Test_RegisterElements.cpp
  Observers/Test_Tags.cpp
  Observers/Test_ObservationId.cpp
  Observers/Test_ReductionObserver.cpp
  Observers/Test_TypeOfObservation.cpp
  Observers/Test_VolumeObserver.cpp
  Test_H5.cpp
  Test_StellarCollapseEos.cpp
  Test_VolumeData.cpp
  )

add_test_library(
  ${LIBRARY}
  "IO"
  "${LIBRARY_SOURCES}"
  "IO"
  )

spectre_add_python_test(
  "Unit.IO.H5.Python"
  "Test_H5.py"
  "unit;IO;H5;python")
