# Distributed under the MIT License.
# See LICENSE.txt for details.

# Allow disabling the Git hooks because if you are running the code in a
# container the host may not have all the right things installed.
option(
  USE_GIT_HOOKS
  "Set up the git hooks for sanity checks."
  ON)

if(USE_GIT_HOOKS AND EXISTS ${CMAKE_SOURCE_DIR}/.git)
  find_package(Git)

  # Hooks are shared between worktrees, so ask git for the repository dir.
  # Not `--git-path hooks`: that follows `core.hooksPath` out of the repo.
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --git-common-dir
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE GIT_COMMON_DIR_RESULT
    OUTPUT_VARIABLE GIT_COMMON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    )
  get_filename_component(GIT_COMMON_DIR "${GIT_COMMON_DIR}" ABSOLUTE
    BASE_DIR ${CMAKE_SOURCE_DIR})

  # Check that the repository dir is writable. If it is we set up git hooks,
  # if not then there probably won't be any commits anyway...
  execute_process(COMMAND test -w "${GIT_COMMON_DIR}"
    RESULT_VARIABLE CHECK_GIT_DIR_WRITABLE_RESULT)

  # The logic is inverted because shell
  if(GIT_COMMON_DIR_RESULT EQUAL 0 AND NOT CHECK_GIT_DIR_WRITABLE_RESULT)
    find_package(ClangFormat)
    set(GIT_HOOKS_DIR ${GIT_COMMON_DIR}/hooks)

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
