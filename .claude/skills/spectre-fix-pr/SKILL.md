---
name: spectre-fix-pr
description: >
  Fetch PR review comments. Trigger when users reference a PR
  (e.g. "address comments on PR #1234", "work on PR #1234 reviews").
allowed-tools: ["Bash", "Read"]
argument-hint: "PR_NUMBER [--repo OWNER/REPO]"
---

Read and follow instructions in file
`$(git rev-parse --show-toplevel)/.skills/spectre-fix-pr/SKILL.md`
