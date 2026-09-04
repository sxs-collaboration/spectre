#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

SPECTRE_DEFAULT_SPEC_ROOT=/resnick/groups/sxs/software/spec/2025-10
SPECTRE_SHARED_PYTHON_VENV=/resnick/groups/sxs/spectre-venv/current
SPECTRE_SHARED_CHARM_ROOT=/groups/sxs/software/charm/8.0.1-gcc-13.2.0-impi-2021.10

spectre_load_required_module() {
    if ! module load "$1" >/dev/null 2>&1; then
        echo "Error: failed to load required module: $1"
        return 1
    fi
}

spectre_load_modules() {
    # Keep core system tools reachable even if node PATH defaults are minimal.
    export PATH=/usr/bin:/bin:$PATH

    module use /central/groups/sxs/modules/
    module use /resnick/groups/sxs/modules/
    module use /central/groups/sxs/spec-modules/modules/
    module use /resnick/groups/sxs/spec-modules/modules/

    # Remove a potentially inherited OpenMPI module before loading Intel MPI.
    module unload openmpi >/dev/null 2>&1 || true
    module unload spec/2025-10 >/dev/null 2>&1 || true

    spectre_load_required_module cmake/3.27.9-gcc-13.2.0-g5dukmj || return 1
    spectre_load_required_module gcc/13.2.0-gcc-13.2.0-w55nxkl || return 1
    spectre_load_required_module intel-oneapi-mpi/2021.10.0-gcc-11.3.1-xj5ixri || return 1
    spectre_load_required_module openblas/0.3.23-gcc-13.2.0-3csynno || return 1
    spectre_load_required_module gsl/2.7.1-gcc-13.2.0-6hld2nf || return 1
    spectre_load_required_module charm/8.0.1-gcc-13.2.0-impi-2021.10 || return 1
    spectre_load_required_module yaml-cpp/0.7.0-gcc-13.2.0-szs4phw || return 1
    spectre_load_required_module libxsmm/1.17-gcc-13.2.0-dnwceqe || return 1
    spectre_load_required_module xsimd/13.2.0 || return 1
    spectre_load_required_module fftw/3.3.10-gcc-11.3.1-ue6oddl || return 1
    spectre_load_required_module patch/2.7.6-gcc-11.3.1-yv35i3a || return 1
    spectre_load_required_module ninja/1.11.1-oneapi-2023.2.1-7jtuoh6   || return 1

    # Optional dependency for visualization support.
    module load paraview/6.0.1-osmesa-py3.10-intelmpi-2021.10.0 >/dev/null 2>&1 || true

    # Avoid UCX issues on this system:
    # https://github.com/sxs-collaboration/spectre/issues/3886
    export FI_PROVIDER=tcp

    if [ -z "${I_MPI_ROOT:-}" ]; then
        echo "Error: I_MPI_ROOT is unset after loading Intel MPI module."
        return 1
    fi
    export SPECTRE_MPI_C_COMPILER="${I_MPI_ROOT}/bin/mpicc"
    export SPECTRE_MPI_CXX_COMPILER="${I_MPI_ROOT}/bin/mpicxx"
    if [ -x "${I_MPI_ROOT}/bin/mpifort" ]; then
        export SPECTRE_MPI_Fortran_COMPILER="${I_MPI_ROOT}/bin/mpifort"
    elif [ -x "${I_MPI_ROOT}/bin/mpif90" ]; then
        export SPECTRE_MPI_Fortran_COMPILER="${I_MPI_ROOT}/bin/mpif90"
    elif [ -x "${I_MPI_ROOT}/bin/mpiifort" ]; then
        export SPECTRE_MPI_Fortran_COMPILER="${I_MPI_ROOT}/bin/mpiifort"
    elif [ -x "${I_MPI_ROOT}/bin/mpiifx" ]; then
        export SPECTRE_MPI_Fortran_COMPILER="${I_MPI_ROOT}/bin/mpiifx"
    else
        echo "Error: no MPI Fortran wrapper found under I_MPI_ROOT:"
        echo "  ${I_MPI_ROOT}/bin"
        return 1
    fi
    export SPECTRE_MPIEXEC_EXECUTABLE="${I_MPI_ROOT}/bin/mpiexec"
    export SPECTRE_MPIRUN_EXECUTABLE="${I_MPI_ROOT}/bin/mpirun"
    export PATH="${I_MPI_ROOT}/bin:${PATH}"

    if [ ! -x "${SPECTRE_MPI_C_COMPILER}" ] || [ ! -x "${SPECTRE_MPI_CXX_COMPILER}" ] || [ ! -x "${SPECTRE_MPI_Fortran_COMPILER}" ] || [ ! -x "${SPECTRE_MPIEXEC_EXECUTABLE}" ] || [ ! -x "${SPECTRE_MPIRUN_EXECUTABLE}" ]; then
        echo "Error: failed to configure Intel MPI wrappers."
        echo "Current values:"
        echo "  SPECTRE_MPI_C_COMPILER=${SPECTRE_MPI_C_COMPILER:-}"
        echo "  SPECTRE_MPI_CXX_COMPILER=${SPECTRE_MPI_CXX_COMPILER:-}"
        echo "  SPECTRE_MPI_Fortran_COMPILER=${SPECTRE_MPI_Fortran_COMPILER:-}"
        echo "  SPECTRE_MPIEXEC_EXECUTABLE=${SPECTRE_MPIEXEC_EXECUTABLE:-}"
        echo "  SPECTRE_MPIRUN_EXECUTABLE=${SPECTRE_MPIRUN_EXECUTABLE:-}"
        return 1
    fi

    # Avoid stale paths from legacy spec-modules modulefiles.
    export BLAZE_ROOT=/resnick/groups/sxs/spec-modules/libraries/blaze/3.8
    export JEMALLOC_ROOT=/resnick/groups/sxs/spec-modules/libraries/jemalloc/5.3.0
    export BOOST_ROOT=/resnick/groups/sxs/spec-modules/libraries/boost/1.82.0
    export HDF5_ROOT=/resnick/groups/sxs/spec-modules/libraries/hdf5/1.12.3
    export SPEC_ROOT="${SPECTRE_DEFAULT_SPEC_ROOT}"
    export CHARM_ROOT="${SPECTRE_SHARED_CHARM_ROOT}"

    if [ -n "${VIRTUAL_ENV:-}" ] && [ "${VIRTUAL_ENV}" != "${SPECTRE_SHARED_PYTHON_VENV}" ] && command -v deactivate >/dev/null 2>&1; then
        deactivate >/dev/null 2>&1 || true
    fi
    if [ ! -f "${SPECTRE_SHARED_PYTHON_VENV}/bin/activate" ]; then
        echo "Error: shared Python venv activate script is missing:"
        echo "  ${SPECTRE_SHARED_PYTHON_VENV}/bin/activate"
        return 1
    fi
    VIRTUAL_ENV_DISABLE_PROMPT=1
    export VIRTUAL_ENV_DISABLE_PROMPT
    . "${SPECTRE_SHARED_PYTHON_VENV}/bin/activate"
    unset VIRTUAL_ENV_DISABLE_PROMPT
    export SPECTRE_PYTHON_EXECUTABLE="${SPECTRE_SHARED_PYTHON_VENV}/bin/python"
    unset PYTHONHOME

    # Avoid Charm trying to rewrite read-only MPIOPTS in shared installs.
    unset MPICXX
    unset MPICC

}

spectre_unload_modules() {
    module use /central/groups/sxs/modules/
    module use /resnick/groups/sxs/modules/
    module use /central/groups/sxs/spec-modules/modules/
    module use /resnick/groups/sxs/spec-modules/modules/

    module unload fftw/3.3.10-gcc-11.3.1-ue6oddl >/dev/null 2>&1 || true
    module unload xsimd/13.2.0 >/dev/null 2>&1 || true
    module unload libxsmm/1.17-gcc-13.2.0-dnwceqe >/dev/null 2>&1 || true
    module unload yaml-cpp/0.7.0-gcc-13.2.0-szs4phw >/dev/null 2>&1 || true
    module unload charm/8.0.1-gcc-13.2.0-impi-2021.10 >/dev/null 2>&1 || true
    module unload gsl/2.7.1-gcc-13.2.0-6hld2nf >/dev/null 2>&1 || true
    module unload openblas/0.3.23-gcc-13.2.0-3csynno >/dev/null 2>&1 || true
    module unload paraview/6.0.1-osmesa-py3.10-intelmpi-2021.10.0 >/dev/null 2>&1 || true
    module unload intel-oneapi-mpi/2021.10.0-gcc-11.3.1-xj5ixri >/dev/null 2>&1 || true
    module unload gcc/13.2.0-gcc-13.2.0-w55nxkl >/dev/null 2>&1 || true
    module unload cmake/3.27.9-gcc-13.2.0-g5dukmj >/dev/null 2>&1 || true

    unset FI_PROVIDER
    unset BOOST_ROOT
    unset HDF5_ROOT
    unset BLAZE_ROOT
    unset JEMALLOC_ROOT
    unset SPEC_ROOT
    unset SPECTRE_MPI_C_COMPILER
    unset SPECTRE_MPI_CXX_COMPILER
    unset SPECTRE_MPI_Fortran_COMPILER
    unset SPECTRE_MPIEXEC_EXECUTABLE
    unset SPECTRE_MPIRUN_EXECUTABLE
    unset SPECTRE_PYTHON_EXECUTABLE
    unset VIRTUAL_ENV
}

spectre_run_cmake() {
    if [ -z "${SPECTRE_HOME:-}" ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    # ParaView is optional in SpECTRE.
    if command -v pvpython >/dev/null 2>&1; then
        enable_paraview=ON
    else
        enable_paraview=OFF
    fi

    # Notes:
    # - Choosing JEMALLOC is important for stability in long BBH runs.
    # - We turn off docs because we aren't loading Doxygen.
    # - We override architecture to skylake for compatibility across nodes.
    cmake -U MPI_* \
          -D CMAKE_C_COMPILER=gcc \
          -D CMAKE_CXX_COMPILER=g++ \
          -D CMAKE_Fortran_COMPILER=gfortran \
          ${SPECTRE_MPI_C_COMPILER:+-D MPI_C_COMPILER=${SPECTRE_MPI_C_COMPILER}} \
          ${SPECTRE_MPI_CXX_COMPILER:+-D MPI_CXX_COMPILER=${SPECTRE_MPI_CXX_COMPILER}} \
          ${SPECTRE_MPI_Fortran_COMPILER:+-D MPI_Fortran_COMPILER=${SPECTRE_MPI_Fortran_COMPILER}} \
          ${SPECTRE_MPIEXEC_EXECUTABLE:+-D MPIEXEC_EXECUTABLE=${SPECTRE_MPIEXEC_EXECUTABLE}} \
          -D CHARM_ROOT="${CHARM_ROOT}" \
          -D BLA_VENDOR=OpenBLAS \
          -D CMAKE_BUILD_TYPE=Release \
          -D BUILD_DOCS=OFF \
          -D DEBUG_SYMBOLS=ON \
          -D MEMORY_ALLOCATOR=JEMALLOC \
          -D BUILD_PYTHON_BINDINGS=ON \
          ${SPEC_ROOT:+-D SPEC_ROOT=${SPEC_ROOT}} \
          -D MACHINE=CaltechHpc \
          -D OVERRIDE_ARCH=skylake \
          -D ENABLE_PARAVIEW="${enable_paraview}" \
          -D BUILD_SHARED_LIBS=ON \
          -D CHARM_SHARED_LIBS=ON \
          -D SPECTRE_FETCH_MISSING_DEPS=ON \
          ${SPECTRE_PYTHON_EXECUTABLE:+-D Python_EXECUTABLE=${SPECTRE_PYTHON_EXECUTABLE}} \
          "$@" \
          "${SPECTRE_HOME}" \
          -G Ninja
}
