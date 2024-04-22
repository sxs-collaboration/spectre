# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(SpEC)

if (NOT SpEC_FOUND)
  return()
endif()

# Make SpEC scripts available in Python. These can be used until we have ported
# them to SpECTRE.
set(PYTHONPATH "${SPEC_ROOT}/Support/Python:${PYTHONPATH}")

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "SpEC exporter: ${SPEC_EXPORTER_ROOT}\n"
  )
