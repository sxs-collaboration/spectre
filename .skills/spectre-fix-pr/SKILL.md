Run the fetch script to retrieve PR metadata, review comments, and changed
files:

```
python3 \
  "$(git rev-parse --show-toplevel)/.skills/scripts/FetchPrComments.py" \
  <number> [--repo owner/repo]
```

- Extract the PR number from the user's message.
- Pass `--repo owner/repo` if the user specifies a repo or says "upstream"
  (for SpECTRE, upstream is `sxs-collaboration/spectre`).

After fetching the PR data:

1. **Summarize the review** before starting work: list the total number of
   threads, how many are unresolved, how many are resolved/outdated if that
   data is available, and the files affected.
2. **Prioritize UNRESOLVED threads** -- these are the comments that still need
   to be addressed. Start with unresolved, non-outdated threads first.
3. **Group by file** -- work through comments file-by-file rather than jumping
   around, since multiple threads often apply to the same file.
4. **Use the diff hunk context** -- each inline comment includes the surrounding
   diff hunk so you can understand exactly what code the reviewer is referring
   to.
5. **Check OUTDATED threads** -- these refer to code that has since changed.
   Verify whether the concern still applies before making changes.
6. **Learn repeated themes before editing** -- if reviewers repeat the same
   concern (formatting churn, missing includes, weak tests, loose tolerances,
   allocations, unclear docs, co-author trailers), check for the same issue
   elsewhere in the PR before calling the feedback addressed.
7. **Skip resolved threads** unless the user specifically asks about them. If
   the user asks for all feedback or a summary, include resolved and outdated
   threads as well.
8. **Before final response**, state which comments were addressed, which are
   obsolete because the code changed, and which remain intentionally unresolved.
