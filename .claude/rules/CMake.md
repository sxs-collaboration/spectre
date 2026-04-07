---
triggers:
  - glob: "**/CMakeLists.txt"
---

- Files listed in spectre_target_sources() and spectre_target_headers() must be
  in alphabetical order (case-insensitive). When adding or removing entries,
  maintain the sorted order.
- Use ${LIBRARY} variable in CMake target calls, not hardcoded library names.
