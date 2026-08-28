#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Runs a short binary black hole simulation and checks its output. This is the
# regression test that the CSCS continuous integration pipeline runs on the
# Eiger cluster (see `.cscs-ci/Eiger.yml` and `docs/DevGuide/CscsCi.md`), but it
# can be run in any environment where the `spectre` CLI, `SolveXcts` and
# `EvolveGhBinaryBlackHole` are in the PATH:
#
#   .cscs-ci/RunBbhRegression.sh /path/to/output/dir
#
# The following environment variables tune the test:
#
#   SPECTRE_NUM_PROCS   Number of Charm++ worker threads (default: 4)
#   SPECTRE_FINAL_TIME  Time in units of M at which to stop (default: 10.0)
#   SPECTRE_ID_L        h-refinement of the initial data solve (default: 0)
#   SPECTRE_ID_P        p-refinement of the initial data solve (default: 8)
#   SPECTRE_EV_L        h-refinement of the evolution (default: 0)
#   SPECTRE_EV_P        p-refinement of the evolution (default: 8)

set -euo pipefail

OUT_DIR=${1:?"Usage: RunBbhRegression.sh OUTPUT_DIR"}
NUM_PROCS=${SPECTRE_NUM_PROCS:-4}
FINAL_TIME=${SPECTRE_FINAL_TIME:-10.0}
ID_L=${SPECTRE_ID_L:-0}
ID_P=${SPECTRE_ID_P:-8}
EV_L=${SPECTRE_EV_L:-0}
EV_P=${SPECTRE_EV_P:-8}

# The final time ends up in the input file as-is, and the option parser rejects
# an integer where it expects a double, so make sure it has a decimal point.
case "${FINAL_TIME}" in
    *.*) ;;
    *) FINAL_TIME="${FINAL_TIME}.0" ;;
esac

ID_DIR=${OUT_DIR}/Id
EV_DIR=${OUT_DIR}/Inspiral
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "${OUT_DIR}"

# Keep a copy of everything as an artifact. Without a scheduler the executables
# stream directly to stdout, they don't write a log file themselves.
exec > >(tee "${OUT_DIR}/RunBbhRegression.log") 2>&1
TEE_PID=$!

# Dump the reduction data as text so it can be inspected, even when the run
# failed. The H5 files themselves are too large to keep as CI artifacts.
extract_dat() {
    local h5_file dat_dir
    for h5_file in "${ID_DIR}"/BbhReductions.h5 "${ID_DIR}"/Horizons.h5 \
                   "${EV_DIR}"/BbhReductions.h5 "${EV_DIR}"/BbhSurfaces.h5; do
        [ -f "${h5_file}" ] || continue
        # One directory per H5 file: `extract-dat -f` deletes the output
        # directory before writing to it.
        dat_dir="$(dirname "${h5_file}")/dat"
        mkdir -p "${dat_dir}"
        spectre extract-dat -f "${h5_file}" \
            "${dat_dir}/$(basename "${h5_file}" .h5)" || true
    done
}
finish() {
    extract_dat
    # Close our end of the pipe and let `tee` drain it, otherwise the tail of
    # the log file can be lost when the script exits.
    exec 1>&- 2>&-
    wait "${TEE_PID}" 2>/dev/null || true
}
trap finish EXIT

echo "=== Generating initial data (L=${ID_L}, P=${ID_P}) ==="
# The `Next` entrypoint in the input file metadata finds the two apparent
# horizons afterwards and writes them to `Horizons.h5`. The initial-data
# control loop is disabled because it would repeat the elliptic solve many
# times.
spectre bbh generate-id \
    --mass-ratio 1 --chi-A 0 0 0 --chi-B 0 0 0 \
    --separation 20 \
    --orbital-angular-velocity 0.01 \
    --radial-expansion-velocity=-1.0e-5 \
    --refinement-level "${ID_L}" --polynomial-order "${ID_P}" \
    --no-control \
    --run-dir "${ID_DIR}" \
    --no-schedule --num-procs "${NUM_PROCS}" --force

echo "=== Evolving to t = ${FINAL_TIME} M (L=${EV_L}, P=${EV_P}) ==="
# `FinalTime` adds a `Completion` event to the input file, see
# `support/Pipelines/Bbh/Inspiral.yaml`. It must be a float: an integer is
# passed through to the input file as an integer, which the option parser
# rejects for a `double`.
#
# We start the inspiral in a separate step rather than passing `--evolve`
# above, because `--evolve` also enables the ringdown continuation (which needs
# a common horizon that a short inspiral never forms), and because the `--lev`
# option it uses cannot select an h-refinement below 1.
spectre bbh start-inspiral "${ID_DIR}/InitialData.yaml" \
    --id-run-dir "${ID_DIR}" --id-subfile-name VolumeData \
    --refinement-level "${EV_L}" --polynomial-order "${EV_P}" \
    --param "FinalTime=${FINAL_TIME}" \
    --run-dir "${EV_DIR}" \
    --no-schedule --num-procs "${NUM_PROCS}" --force

echo "=== Checking output ==="
python-spectre "${SCRIPT_DIR}/CheckBbhRegression.py" \
    --id-dir "${ID_DIR}" \
    --inspiral-dir "${EV_DIR}" \
    --final-time "${FINAL_TIME}" \
    --metrics-output "${OUT_DIR}/performance.json"
