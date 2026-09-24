---
name: spectre-pr-ready
description: >
  Run a SpECTRE pre-PR readiness check on local changes before opening a PR or
  marking it ready for review.
allowed-tools: ["Bash", "Read", "Grep", "Glob"]
argument-hint: "[clang-tidy] [coverage]"
---

Read and follow instructions in file
`$(git rev-parse --show-toplevel)/.skills/spectre-pr-ready/SKILL.md`
