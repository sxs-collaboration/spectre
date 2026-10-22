# Distributed under the MIT License.
# See LICENSE.txt for details.

# Set up position independent code by default since this is required
# for our python libraries.
#
# This adds `-fPIC` to the compile flags. It overrides any `-fPIE/-fPIC` flags
# that CMake's set(CMAKE_POSITION_INDEPENDENT_CODE ON) adds, so all targets have
# consistent flags and we can use one PCH for all of them.
set_property(TARGET SpectreFlags
  APPEND PROPERTY
  INTERFACE_COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:C>:-fPIC>
  $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
  $<$<COMPILE_LANGUAGE:Fortran>:-fPIC>)
