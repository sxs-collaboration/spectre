# Distributed under the MIT License.
# See LICENSE.txt for details.

# Allow disabling the Git hooks because if you are running the code in a
# container the host may not have all the right things installed.
option(
  USE_GIT_HOOKS
  "Set up the git hooks for sanity checks."
  ON)

find_package(Git)

if(USE_GIT_HOOKS AND Git_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/.git)
  find_package(ClangFormat)

  # Hooks are shared between worktrees, so ask git where they live
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --git-path hooks
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HOOKS_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  get_filename_component(GIT_HOOKS_DIR "${GIT_HOOKS_DIR}" ABSOLUTE
    BASE_DIR ${CMAKE_SOURCE_DIR})

  # Check that the hooks dir is writable. If it is we set up git hooks, if not
  # then there probably won't be any commits anyway...
  execute_process(COMMAND test -w ${GIT_HOOKS_DIR}
    RESULT_VARIABLE CHECK_HOOKS_DIR_WRITABLE_RESULT)

  # The logic is inverted because shell
  if(NOT CHECK_HOOKS_DIR_WRITABLE_RESULT)
    # We use several client-side git hooks to ensure commits are correct as
    # early as possible.
    configure_file(
      ${CMAKE_SOURCE_DIR}/tools/Hooks/pre-commit.sh
      ${GIT_HOOKS_DIR}/pre-commit
      @ONLY
      )
    configure_file(
      ${CMAKE_SOURCE_DIR}/tools/Hooks/CheckFileSize.py
      ${GIT_HOOKS_DIR}/CheckFileSize.py
      @ONLY
      )
    configure_file(
      ${CMAKE_SOURCE_DIR}/tools/Hooks/post-checkout.sh
      ${GIT_HOOKS_DIR}/post-checkout
      COPYONLY
      )
  endif()
endif()
