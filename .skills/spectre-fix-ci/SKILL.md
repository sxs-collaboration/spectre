Run the fetch script to retrieve and analyze the CI job log:

```
python3 \
  "$(git rev-parse --show-toplevel)/.skills/scripts/FetchCiLog.py" \
  <JOB_ID> [--owner OWNER] [--context LINES]
```

- Extract the job ID from the user's message (required).
- Pass `--owner OWNER` if the user specifies an owner (default is
  `sxs-collaboration`).
- `--context` defaults to 50 lines before and after each failure.

After the script returns:

1. **Summarize the failure**: which step failed, which test(s) failed,
   and what the error was.
2. **Identify source locations**: find file paths and line numbers in the
   error output that point to the failing code.
3. **Classify the failure**: compile, unit-test assertion, timeout,
   infrastructure/dependency, formatting, clang-tidy, docs, or unknown.
4. **Timeouts**: if tests only timed out, rerun each timed-out test serially
   before changing code.
5. **Tolerance failures**: investigate numerical stability or real `src` bugs
   before relaxing tolerances.
6. **Suggest next steps**: propose a fix or further investigation. If the
   failure looks unrelated to the PR, say why and identify the component that
   failed.

If the context window seems too narrow to understand the failure, re-run
with `--context 150` for more surrounding lines.
