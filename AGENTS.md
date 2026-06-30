# Project structure
- Source code is in `src/`, tests in `tests/`, documentation in `docs/`.
- The main branch is `develop`.

# Configure and build
- Use the CMake build system. Place build directories in the repo named `build*`
  (git-ignored).
- Prefer CMake presets listed in CMakePresets.json and CMakeUserPresets.json.
  Use `debug` for most work; use `release-debug` for performance-sensitive work
  (optimized but with sanity checks). Build directories are named
  `build-<preset_name>`. If presets don't exist, suggest creating
  CMakeUserPresets.json following support/Environments/, or fall back to
  `-D CMAKE_BUILD_TYPE=Debug`. Prefer clang over gcc if both are available. More
  options in docs/Installation/BuildSystem.md.
- In a git worktree, symlink the git-ignored CMakeUserPresets.json from the main
  checkout and pass `-D USE_GIT_HOOKS=OFF` to CMake. Suggest adding a
  `post-checkout` git hook in `.git/hooks` to create the symlink automatically.
- Never build `all`. Build only the targets you need (find them in the closest
  CMakeLists.txt).

# Testing
- Run tests with `ctest --output-on-failure`.
- Run only tests affected by changes for fast feedback. To catch regressions,
  build `unit-tests` and run `ctest -L unit -j <number_of_cores>` (set
  number_of_cores from AGENTS.local.md, otherwise use 2).

# Commit & PR guidelines
- Use short, imperative subject lines and reference issue numbers when
  available. PRs summarize motivation, user-visible effects, and testing
  performed.

# Running executables and scripts
- Input files are YAML (see how to run executables in the InputFiles rule).
- `<build_directory>/bin/spectre` is a self-documenting CLI: use `--help` to
  list subcommands and options. Build the `cli` target to enable it.
- Run Python with `<build_directory>/bin/python-spectre`, never plain `python`.
  If import fails, build `all-pybindings`. The `spectre` package mirrors the C++
  tree (`src/<Path>/Python/` -> `spectre.<Path>`).

# Additional instructions
- Always prefer personal overrides: @AGENTS.local.md
