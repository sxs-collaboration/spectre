Run the fetch script to retrieve the issue title, body, and comments:

```
python3 \
  "$(git rev-parse --show-toplevel)/.skills/scripts/FetchIssue.py" \
  <number> [--repo owner/repo]
```

- Extract the issue number from the user's message.
- Pass `--repo owner/repo` if the user specifies a repo or says "upstream"
  (for SpECTRE, upstream is `sxs-collaboration/spectre`).
- Summarize the issue, then proceed with the user's request.
