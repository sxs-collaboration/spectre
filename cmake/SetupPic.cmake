# Distributed under the MIT License.
# See LICENSE.txt for details.

# Set up position independent code by default since this is required
# for our python libraries.
#
# We cannot use CMake's set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# because that enables -fPIC in shared libraries and -fPIE on executables.
# While what CMake does is technically the optimal thing to do, it means
# we would need to generate two PCHs, one built with -fPIC and one with
# -fPIE. More specifically, we'd need to generate one using a library target
# and one using an executable target, which would double the amount of code
# in SetupPch.cmake for a currently unknown benefit.
set_property(TARGET SpectreFlags
  APPEND PROPERTY
  INTERFACE_COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:CXX>:-fPIC>)
