#!/usr/bin/env sh
# Distributed under the MIT License.
# See LICENSE.txt for details.
#
# Self-locating launcher for the SpECTRE MCP server, shared by all MCP clients
# (Claude Code, GitHub Copilot, OpenAI Codex, Mistral Vibe). Each client's MCP
# config points its 'command' at this script. The script resolves the
# repository root from its OWN path, so it does not depend on any client's
# working directory or on a client-specific variable such as
# 'CLAUDE_PROJECT_DIR'. Extra arguments are forwarded to 'spectre mcp'.
#
# The build directory defaults to '<repo>/build' (the repo convention) and can
# be overridden with the SPECTRE_BUILD_DIR environment variable.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(cd -- "$script_dir/../.." && pwd -P)
spectre="${SPECTRE_BUILD_DIR:-$repo_root/build}/bin/spectre"

if [ ! -x "$spectre" ]; then
  echo "spectre CLI not found or not executable at: $spectre" >&2
  echo "Build SpECTRE, or set SPECTRE_BUILD_DIR to your build directory." >&2
  exit 1
fi

exec "$spectre" mcp "$@"
