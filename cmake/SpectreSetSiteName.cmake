# Distributed under the MIT License.
# See LICENSE.txt for details.

site_name(HOSTNAME)

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "Hostname: ${HOSTNAME}\n"
  "Host system: ${CMAKE_HOST_SYSTEM}\n"
  "Host system version: ${CMAKE_HOST_SYSTEM_VERSION}\n"
  "Host system processor: ${CMAKE_HOST_SYSTEM_PROCESSOR}\n"
  )
