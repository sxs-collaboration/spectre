Run the fetch script to retrieve the issue title, body, and comments:

```
python3 \
  "$(git rev-parse --show-toplevel)/.skills/scripts/FetchIssue.py" \
  <number> [--repo owner/repo] [--full]
```

- Extract the issue number from the user's message.
- Pass `--repo owner/repo` if the user specifies a repo or says "upstream"
  (for SpECTRE, upstream is `sxs-collaboration/spectre`).
- Pass `--full` if the user explicitly asks for the full/unshortened issue
  text (e.g. "show me the full issue", "don't shorten it", "skip random
  failure handling"). Do NOT pass `--full` by default.

**If the output starts with `[RANDOM_FAILURE]`**: Read and follow the
instructions in
`$(git rev-parse --show-toplevel)/.skills/spectre-fix-issue/RANDOM_FAILURE.md`
using the seed data from the script output. Mention to the user that they can
ask for the full issue text if they need it.

**Otherwise**: Summarize the issue, then proceed with the user's request.
