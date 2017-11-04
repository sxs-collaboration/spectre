# Distributed under the MIT License.
# See LICENSE.txt for details.

# The target is used for debugging preprocessor metaprograms by
# outputting the preprocessed source file to stdout

# Get a list of the include dirs, then prepend -I to all of them
get_property(
  DEBUG_PREPROCESSOR_INCLUDE_DIRS
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY INCLUDE_DIRECTORIES
  )
string(
  REGEX REPLACE ";" ";-I"
  DEBUG_PREPROCESSOR_INCLUDE_DIRS
  "${DEBUG_PREPROCESSOR_INCLUDE_DIRS}"
  )
set(DEBUG_PREPROCESSOR_INCLUDE_DIRS
  "-I${DEBUG_PREPROCESSOR_INCLUDE_DIRS}"
  )

add_custom_target(
  DebugPreprocessor
  COMMAND ${CMAKE_CXX_COMPILER}
  ${DEBUG_PREPROCESSOR_INCLUDE_DIRS}
  -std=c++14 # needed to get output
  ${CMAKE_SOURCE_DIR}/src/Executables/DebugPreprocessor/DebugPreprocessor.cpp
  -E
  )
