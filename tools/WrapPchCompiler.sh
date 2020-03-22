#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

compiler_invokation=("$@")

pch_args=()
while [ $# -ne 0 ] ; do
  case "$1" in
    -c|-o)
      shift 2
      ;;
    *)
      pch_args+=("$1")
      shift
      ;;
  esac
done

"${pch_args[@]}" @SPECTRE_PCH_HEADER_PATH@ -o @SPECTRE_PCH_PATH@

"${compiler_invokation[@]}"
