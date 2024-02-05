# Distributed under the MIT License.
# See LICENSE.txt for details.

option(ENABLE_OPENMP "Enable OpenMP in some parts of the code" OFF)

if (ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
endif()
