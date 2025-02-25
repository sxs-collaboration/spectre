# Distributed under the MIT License.
# See LICENSE.txt for details.

list(APPEND
  CMAKE_PREFIX_PATH
  /opt/sxs/software/gcc-11.4.0/blaze/3.8
  /opt/sxs/software/gcc-11.4.0/boost/1.82.0
  /opt/sxs/software/gcc-11.4.0/catch/3.5.1
  /opt/sxs/software/gcc-11.4.0/ccache/4.9
  /opt/sxs/software/gcc-11.4.0/charm/7.0.0
  /opt/sxs/software/gcc-11.4.0/doxygen/1.10.0
  /opt/sxs/software/gcc-11.4.0/fftw/3.3.10
  /opt/sxs/software/gcc-11.4.0/google_benchmark/1.8.3
  /opt/sxs/software/gcc-11.4.0/gsl/2.7
  /opt/sxs/software/gcc-11.4.0/hdf5/1.12.3
  /opt/sxs/software/gcc-11.4.0/intel/mpi/2021.11/mpi/2021.11
  /opt/sxs/software/gcc-11.4.0/intel/mpi/2021.11/mpi/2021.11/opt/mpi/libfabric
  /opt/sxs/software/gcc-11.4.0/jemalloc/5.3.0
  /opt/sxs/software/gcc-11.4.0/libbacktrace/1.0
  /opt/sxs/software/gcc-11.4.0/libxsmm/1.16.1
  /opt/sxs/software/gcc-11.4.0/ninja/1.10.1
  /opt/sxs/software/gcc-11.4.0/openblas/0.3.25
  /opt/sxs/software/gcc-11.4.0/paraview/5.11.1
  /opt/sxs/software/gcc-11.4.0/petsc/3.13.6
  /opt/sxs/software/gcc-11.4.0/spectre-python/3.10.3
  /opt/sxs/software/gcc-11.4.0/xsimd/12.1.1
  /opt/sxs/software/gcc-11.4.0/yaml-cpp/0.8.0
  )
set(CHARM_ROOT /opt/sxs/software/gcc-11.4.0/charm/7.0.0 CACHE PATH "")
set(USE_XSIMD ON CACHE  BOOL "")
set(SPEC_ROOT /opt/sxs/software/gcc-11.4.0/spec-exporter/2024-02-06
  CACHE PATH "")
set(ENABLE_PARAVIEW ON CACHE BOOL "")
set(BUILD_DOCS ON CACHE BOOL "")
set(BUILD_PYTHON_BINDINGS ON CACHE BOOL "")
set(MACHINE "Mbot" CACHE STRING "")
set(SPECTRE_MPI_LAUNCHER "mpirun" CACHE STRING "")
