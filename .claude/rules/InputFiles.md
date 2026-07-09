---
triggers:
  - glob: "**/*.yaml"
---

# Running executables from input files
- Input files specify runtime options for executables in YAML. The corresponding
  executable is named near the top with `Executable: <executable_name>`. Build
  that target, then run it with
  `<build_directory>/bin/<executable_name> --input-file <input_file.yaml>`. Add
  `+p <number_of_cores>` to run in parallel.
- Run executables in a temporary run directory inside the build directory to
  avoid stale output files.
- Pipe output to `spectre.log` for easier debugging:
  `<executable> --input-file <input_file.yaml> | tee spectre.log`.
