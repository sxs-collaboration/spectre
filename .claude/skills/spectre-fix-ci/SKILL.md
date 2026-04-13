---
name: spectre-fix-ci
description: >
  Fetch and analyze a failed GitHub Actions job log. When viewing a job
  in GitHub Actions, the job ID is the last number in the URL, e.g. in
  "runs/24319442407/job/71002790711" the job ID is "71002790711".
allowed-tools: ["Bash"]
argument-hint: "JOB_ID [--owner OWNER]"
user-invocable: true
model-invocable: false
---

Read and follow instructions in file
`$(git rev-parse --show-toplevel)/.skills/spectre-fix-ci/SKILL.md`
