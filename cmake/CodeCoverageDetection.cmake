################################################################################
#
# \file      cmake/CodeCoverageDetection.cmake
# \author    J. Bakosi
# \copyright 2012-2015, Jozsef Bakosi, 2016, Los Alamos National Security, LLC.
# \brief     Detect prerequesites for code coverage analysis
# \date      Fri 03 Mar 2017 11:50:24 AM MST
#
# Modifications for SpECTRE:
# 1) Auto find either llvm-cov or llvm-cov-${LLVM_VERSION} instead of using
#    a shell script that hard codes the LLVM version
# 2) Formatting changes
################################################################################

# Attempt to find tools required for code coverage analysis

if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  string(
      REGEX MATCH "^[0-9]+.[0-9]+" LLVM_VERSION
      "${CMAKE_CXX_COMPILER_VERSION}"
  )
  find_program(
      LLVM_COV_BIN
      NAMES "llvm-cov-${LLVM_VERSION}" "llvm-cov"
      HINTS ${COMPILER_PATH}
  )
  configure_file(
      "${CMAKE_SOURCE_DIR}/tools/llvm-gcov.sh"
      "${CMAKE_BINARY_DIR}/llvm-gcov.sh"
  )
  set(GCOV "${CMAKE_BINARY_DIR}/llvm-gcov.sh")
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
  find_program(GCOV gcov)
endif()

find_program( LCOV lcov )
find_program( GENHTML genhtml )
find_program( SED sed )

# Code coverage analysis only supported if all prerequisites found and the user
# has requested it via the cmake variable COVERAGE=on..
if(COVERAGE AND GCOV AND LCOV AND GENHTML AND SED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

  # Enable code coverage analysis.
  SET(CODE_COVERAGE ON)

  # Make flag enabling code coverage analysis available in parent cmake scope
  mark_as_advanced(CODE_COVERAGE)

  # Only include code coverage cmake functions if all prerequsites are met
  include(CodeCoverage)
elseif(COVERAGE)
  message(FATAL_ERROR "Failed to enable code coverage analysis. Not all "
      "prerequisites found: gcov:${GCOV}, lcov:${LCOV}, genhtml:${GENHTML},"
      " sed:${SED}")
endif()
