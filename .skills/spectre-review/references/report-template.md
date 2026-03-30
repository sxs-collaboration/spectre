# Final Report Template

Present a clean report in the format below. Be brief and specific. No emojis.
Link to files and lines.

```
### SpECTRE Code Review

**Reviewing**: [PR #N: title | local changes on branch `name`]
**Files changed**: N (X C++, Y Python, Z CMake)

#### Formatting
[clang-format / black / isort changes, or "No formatting issues."]

#### Critical Issues
N. **Description** `file:line`
   Explanation and suggested fix.

#### Important Issues
N. **Description** `file:line`
   Explanation.

#### Suggestions
N. **Description** `file:line`
   Explanation.

#### CI Pre-Check Warnings
[Issues that CI will flag, or "None."]

[If clang-tidy ran:]
#### clang-tidy
[Findings or "No issues."]

[If coverage ran:]
#### Code Coverage (changed lines)
[Uncovered changed lines per file, or "All changed lines are covered by tests."]
[Files with no test target: "Coverage not checked: file1.cpp, file2.hpp (no test
target found)"]
```

If no issues at all: "No issues found. Checked style, patterns, bugs,
formatting, and CI compliance."
