# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(ClangFormat)

if(CLANG_FORMAT_BIN AND EXISTS ${CMAKE_SOURCE_DIR}/.git AND Git_FOUND)
  get_filename_component(CLANG_FORMAT_NAME ${CLANG_FORMAT_BIN} NAME)
  add_custom_target(
      git-clang-format
      COMMAND
      export PATH=${CMAKE_SOURCE_DIR}/tools:$$PATH
      && cd ${CMAKE_SOURCE_DIR}
      && ${GIT_EXECUTABLE} ${CLANG_FORMAT_NAME} -f
  )
endif()
