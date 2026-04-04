---
name: spectre-review
description: SpECTRE code review (PR or local commits)
argument-hint: "[PR#] [clang-tidy] [coverage]"
allowed-tools: ["Bash", "Read", "Grep", "Glob", "Agent", "AskUserQuestion", "TaskCreate", "TaskUpdate", "TaskList"]
disable-model-invocation: true
---

Read and follow instructions in file
`$(git rev-parse --show-toplevel)/.skills/spectre-review/SKILL.md`
