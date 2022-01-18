# Distributed under the MIT License.
# See LICENSE.txt for details.

include(CTest)

set(SPECTRE_TEST_RUNNER "" CACHE STRING
  "Run test executables through the given wrapper.")

option(SPECTRE_TEST_TIMEOUT_FACTOR
  "Multiply timeout for tests by this factor")

function(spectre_define_test_timeout_factor_option TEST_TYPE HELP_NAME)
  option(SPECTRE_${TEST_TYPE}_TEST_TIMEOUT_FACTOR
    "Multiply timeout for ${HELP_NAME} tests by this factor")
endfunction ()

# Multiply BASE_TIMEOUT by the user specified option for TEST_TYPE
function(spectre_test_timeout RETURN_VARIABLE TEST_TYPE BASE_TIMEOUT)
  if (SPECTRE_${TEST_TYPE}_TEST_TIMEOUT_FACTOR)
    set(FACTOR ${SPECTRE_${TEST_TYPE}_TEST_TIMEOUT_FACTOR})
  elseif (SPECTRE_TEST_TIMEOUT_FACTOR)
    set(FACTOR ${SPECTRE_TEST_TIMEOUT_FACTOR})
  else ()
    set(FACTOR 1)
  endif ()

  # Note: "1" is parsed as "ON" by cmake
  if (NOT "${FACTOR}" STREQUAL ON)
    math(EXPR RESULT "${FACTOR} * ${BASE_TIMEOUT}")
  else ()
    set(RESULT ${BASE_TIMEOUT})
  endif ()

  set(${RETURN_VARIABLE} ${RESULT} PARENT_SCOPE)
endfunction ()
