# Distributed under the MIT License.
# See LICENSE.txt for details.

set(_LIST_OF_3PLS
  Blaze
  Boost
  Brigand
  Catch2
  Gsl
  Hdf5
  Jemalloc
  Libbacktrace
  Libsharp
  Libxsmm
  OpenBlas
  Pybind11
  Xsimd
  YamlCpp
)

file(READ "${CMAKE_SOURCE_DIR}/LICENSE.txt" SPECTRE_MIT_LICENSE)
file(READ "${CMAKE_SOURCE_DIR}/external/Licenses/SpectreGpl3License.txt"
      SPECTRE_GPL3_LICENSE)
set(SPECTRE_LICENSE_CXX_STRING "
SpECTRE source code is distributed under the MIT License, while SpECTRE
executables and libraries, generally machine code, is distributed under the GNU
Public License v3, or GPL3. The SpECTRE source code is freely available at
https://github.com/sxs-collaboration/spectre/

SpECTRE copyright for source, executables, libraries, and machine code is:
Copyright 2017 - 2024 Simulating eXtreme Spacetimes Collaboration

SpECTRE source code MIT license:
${SPECTRE_MIT_LICENSE}

SpECTRE executable, library, and machine code license:
${SPECTRE_GPL3_LICENSE}




Third Party Libraries:
")

# Note: C++ compilers only need to support 65,000 characters in raw string
# literals, so we split this over 3 strings. 1 for the SpECTRE licenses,
# and two of the 3PLs.
set(SPECTRE_3PL_CXX_STRING0 "")
set(SPECTRE_3PL_CXX_STRING1 "")

set(SPECTRE_3PL_DOX_STRING "${SPECTRE_LICENSE_CXX_STRING}")

foreach(_3PL ${_LIST_OF_3PLS})
  if(EXISTS "${CMAKE_SOURCE_DIR}/external/Licenses/${_3PL}Copyright.txt")
    file(READ "${CMAKE_SOURCE_DIR}/external/Licenses/${_3PL}Copyright.txt"
      _3PL_COPYRIGHT)
  else()
    set(_3PL_COPYRIGHT "")
  endif()
  file(READ "${CMAKE_SOURCE_DIR}/external/Licenses/${_3PL}License.txt"
    _3PL_LICENSE)
  string(COMPARE LESS ${_3PL} "Libbacktrace" _STRING_0_or_1)
  if(_STRING_0_or_1)
    set(SPECTRE_3PL_CXX_STRING0
      "${SPECTRE_3PL_CXX_STRING0}${_3PL}\n${_3PL_COPYRIGHT}\nDistributed under \
the license\n${_3PL_LICENSE}\n\n\n")
  else()
    set(SPECTRE_3PL_CXX_STRING1
      "${SPECTRE_3PL_CXX_STRING1}${_3PL}\n${_3PL_COPYRIGHT}\nDistributed under \
the license\n${_3PL_LICENSE}\n\n\n")
  endif()
  set(SPECTRE_3PL_DOX_STRING
    "${SPECTRE_3PL_DOX_STRING}###${_3PL}\n${_3PL_COPYRIGHT}\nDistributed under \
the license\n```\n${_3PL_LICENSE}\n```\n\n\n")
endforeach()
