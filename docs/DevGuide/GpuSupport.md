\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# GPU support {#gpu_support}

\tableofcontents

# Overview

\warning GPU support is experimental and this page will be updated as the
implementation matures.

SpECTRE supports GPU acceleration through the [Kokkos](https://github.com/kokkos/kokkos)
library.

## Build configuration

To enable GPU support, set the CMake option `-D SPECTRE_KOKKOS=ON` when
configuring the build. Either point CMake to a Kokkos installation with
`-D Kokkos_ROOT=path/to/kokkos` or set `-D SPECTRE_FETCH_MISSING_DEPS=ON` to
fetch Kokkos automatically and build it as part of SpECTRE. You also have to
select a parallelization backend for Kokkos and possibly more configuration
options like the GPU architecture to build for. Read the
[Kokkos documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)
for details on how to configure Kokkos. Here's an example for selecting the CUDA
backend:

```sh
cmake -D SPECTRE_KOKKOS=ON \
      -D Kokkos_ROOT=path/to/kokkos/build \
      -D Kokkos_ENABLE_CUDA=ON \
      -D CMAKE_CXX_COMPILER=path/to/kokkos/bin/nvcc_wrapper \
      ...
```
