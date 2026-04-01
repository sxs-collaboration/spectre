---
name: spectre-fix-issue
description: >
  Fetch a GitHub issue by number. Trigger when the user references a GitHub
  issue (e.g. "fix issue #1727", "work on #1727").
allowed-tools: ["Bash"]
argument-hint: "ISSUE_NUMBER [--repo OWNER/REPO]"
---

Read and follow instructions in file
`$(git rev-parse --show-toplevel)/.skills/spectre-fix-issue/SKILL.md`
