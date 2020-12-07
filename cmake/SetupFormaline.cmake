# Distributed under the MIT License.
# See LICENSE.txt for details.

# When running formaline when SpECTRE was not built from a Git repository
# we need a list of all the files and directories in the root of the source
# directory that are tracked by Git. I.e.
#   git ls-tree --full-tree --name-only HEAD
set(SPECTRE_FORMALINE_LOCATIONS
  ".clang-format;cmake;CMakeLists.txt;containers;docs;external;"
  ".github;.gitignore;LICENSE.txt;Metadata.yaml;README.md;src;.style.yapf;"
  "support;tests;tools;.travis;.travis.yml")

find_package(Git)

if(EXISTS ${CMAKE_SOURCE_DIR}/.git AND Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} ls-tree --full-tree --name-only HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE FORMALINE_GIT_FILES_TRACKED
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  string(REPLACE "\n" ";" FORMALINE_GIT_FILES_TRACKED
    "${FORMALINE_GIT_FILES_TRACKED}")

  # Check all elements in SPECTRE_FORMALINE_LOCATIONS are in
  # FORMALINE_GIT_FILES_TRACKED. We don't just check that the sorted lists
  # are equal because we want to be able to print out exactly which file
  # or directory is missing.
  foreach(FILE ${SPECTRE_FORMALINE_LOCATIONS})
    list(FIND FORMALINE_GIT_FILES_TRACKED ${FILE} FOUND_FILE)
    if(${FOUND_FILE} EQUAL -1)
      message(FATAL_ERROR
        "Couldn't find the file or directory \"${FILE}\" in the "
        "source directory. You need to remove the entry \"${FILE}\" "
        "from the CMake variable SPECTRE_FORMALINE_LOCATIONS in "
        "cmake/SetupFormaline.cmake")
    endif()
  endforeach()

  # Check all elements in FORMALINE_GIT_FILES_TRACKED are in
  # SPECTRE_FORMALINE_LOCATIONS. We don't just check that the sorted lists
  # are equal because we want to be able to print out exactly which file
  # or directory is missing.
  foreach(FILE ${FORMALINE_GIT_FILES_TRACKED})
    list(FIND SPECTRE_FORMALINE_LOCATIONS ${FILE} FOUND_FILE)
    if(${FOUND_FILE} EQUAL -1)
      message(FATAL_ERROR
        "Couldn't find the file or directory \"${FILE}\" in the "
        "CMake variable SPECTRE_FORMALINE_LOCATIONS in "
        "cmake/SetupFormaline.cmake. You need to add \"${FILE}\" "
        "to the CMake variable SPECTRE_FORMALINE_LOCATIONS in "
        "cmake/SetupFormaline.cmake")
    endif()
  endforeach()
endif()

option(USE_FORMALINE
  "Use Formaline to encode the source tree into executables and output files."
  ON)

# APPLE instead of ${APPLE} is intentional
if (APPLE)
  set(USE_FORMALINE OFF)
endif (APPLE)

if (USE_FORMALINE)
  # Create a variable that is space-delimited to insert into the Formaline
  # shell script
  string(REPLACE ";" " " SPECTRE_FORMALINE_LOCATIONS_SHELL
    "${SPECTRE_FORMALINE_LOCATIONS}")

  configure_file(
    ${CMAKE_SOURCE_DIR}/tools/Formaline.sh
    ${CMAKE_BINARY_DIR}/tmp/Formaline.sh
    @ONLY
    )
else()
  file(REMOVE ${CMAKE_BINARY_DIR}/tmp/Formaline.sh)
endif()
