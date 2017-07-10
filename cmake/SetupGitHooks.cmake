# Distributed under the MIT License.
# See LICENSE.txt for details.

# Check that the source dir is writable. If it is we set up git hooks, if not
# then there probably won't be any commits anyway...
EXECUTE_PROCESS(COMMAND test -w ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE CHECK_SOURCE_DIR_WRITABLE_RESULT)

# The logic is inverted because shell
if(NOT CHECK_SOURCE_DIR_WRITABLE_RESULT)
  find_package(PythonInterp REQUIRED)

  find_package(Git REQUIRED)

  # We use several client-side git hooks to ensure commits are correct as
  # early as possible.
  configure_file(
      ${CMAKE_SOURCE_DIR}/tools/Hooks/pre-commit.sh
      ${CMAKE_SOURCE_DIR}/.git/hooks/pre-commit
      @ONLY
  )

  configure_file(
      ${CMAKE_SOURCE_DIR}/tools/Hooks/ClangFormat.py
      ${CMAKE_SOURCE_DIR}/.git/hooks/ClangFormat.py
      @ONLY
  )

  configure_file(
      ${CMAKE_SOURCE_DIR}/tools/Hooks/CheckFileSize.py
      ${CMAKE_SOURCE_DIR}/.git/hooks/CheckFileSize.py
      @ONLY
  )
endif()
