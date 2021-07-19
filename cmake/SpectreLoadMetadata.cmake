# Distributed under the MIT License.
# See LICENSE.txt for details.

# Read the following information from the `Metadata.yaml` file:
# - SPECTRE_NAME: Name
# - SPECTRE_HOMEPAGE: Homepage
# - SPECTRE_VERSION: Version
# - SPECTRE_DOI: Doi
# - SPECTRE_ZENODO_ID: ZenodoId

file(READ "${CMAKE_SOURCE_DIR}/Metadata.yaml" SPECTRE_METADATA_FILE)
# Split lines into cmake list
string(REGEX REPLACE ";" "\\\\;" SPECTRE_METADATA_FILE "${SPECTRE_METADATA_FILE}")
string(REGEX REPLACE "\n" ";" SPECTRE_METADATA_FILE "${SPECTRE_METADATA_FILE}")

# Read data from the file
set(SPECTRE_NAME "")
set(SPECTRE_HOMEPAGE "")
set(SPECTRE_VERSION "")
set(SPECTRE_DOI "")
set(SPECTRE_ZENODO_ID "")
foreach(LINE ${SPECTRE_METADATA_FILE})
  if("${LINE}" MATCHES "^Name: (.*)$")
    set(SPECTRE_NAME ${CMAKE_MATCH_1})
  endif()
  if("${LINE}" MATCHES "^Homepage: (.*)$")
    set(SPECTRE_HOMEPAGE ${CMAKE_MATCH_1})
  endif()
  if("${LINE}" MATCHES "^Version: (.*)$")
    set(SPECTRE_VERSION ${CMAKE_MATCH_1})
  endif()
  if("${LINE}" MATCHES "^Doi: (.*)$")
    set(SPECTRE_DOI ${CMAKE_MATCH_1})
  endif()
  if("${LINE}" MATCHES "^ZenodoId: (.*)$")
    set(SPECTRE_ZENODO_ID ${CMAKE_MATCH_1})
  endif()
endforeach()

# Check we have read all required keys
if(NOT SPECTRE_NAME)
  message(
    FATAL_ERROR "Could not determine project name. Please check the "
    "Metadata.yaml file exists and has a line that matches: ^Name: (.*)$")
endif()
if(NOT SPECTRE_HOMEPAGE)
  message(
    WARNING "Could not determine project homepage. Please check the "
    "Metadata.yaml file exists and has a line that matches: ^Homepage: (.*)$")
endif()
if(NOT SPECTRE_VERSION)
  message(
    FATAL_ERROR "Could not determine release version. Please check the "
    "Metadata.yaml file exists and has a line that matches: ^Version: (.*)$")
endif()
if(NOT SPECTRE_DOI)
  message(
    WARNING "Could not determine DOI. Please check the "
    "Metadata.yaml file exists and has a line that matches: ^Doi: (.*)$")
endif()
if(NOT SPECTRE_ZENODO_ID)
  message(
    WARNING "Could not determine Zenodo ID. Please check the "
    "Metadata.yaml file exists and has a line that matches: ^ZenodoId: (.*)$")
endif()

message(STATUS "${SPECTRE_NAME} release version: ${SPECTRE_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "SpECTRE Version: ${SPECTRE_VERSION}\n"
  )
