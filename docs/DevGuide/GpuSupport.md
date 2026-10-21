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
for details on how to configure Kokkos. Here's an example for fetching Kokkos
automatically and building it as part of SpECTRE with the CUDA backend:

```sh
cmake -D SPECTRE_KOKKOS=ON \
      -D SPECTRE_FETCH_MISSING_DEPS=ON \
      -D Kokkos_ENABLE_CUDA=ON \
      ...
```

Here's an example for using an existing Kokkos installation:

```sh
cmake -D SPECTRE_KOKKOS=ON \
      -D Kokkos_ROOT=path/to/kokkos/build \
      -D CMAKE_CXX_COMPILER=path/to/kokkos/bin/nvcc_wrapper \
      ...
```

When building Kokkos separately with the CUDA backend, you have to set the
following configuration options:

- `Kokkos_ENABLE_CUDA_CONSTEXPR=ON`
- `Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE`

## Writing code for devices

Most of SpECTRE is built without Kokkos, so code for devices must also compile,
and be checked by tools like clang-tidy, in builds without Kokkos:

- Annotate functions that should be callable on the device with
  `KOKKOS_FUNCTION` from `Utilities/Kokkos/KokkosCore.hpp`. The annotation does
  nothing in builds without Kokkos, so no other changes are needed.
- Put code that uses Kokkos types, such as `Kokkos::View` or
  `Kokkos::parallel_for`, in a separate file named after its host counterpart
  with a `Kokkos` suffix, e.g. `VariablesKokkos.hpp`. Wrap everything after the
  includes in `#ifdef SPECTRE_KOKKOS`, which is defined when building with
  Kokkos. Add Kokkos-only tests to the build only `if(SPECTRE_KOKKOS)`. Avoid
  `#ifdef SPECTRE_KOKKOS` branches within files shared with host code.
- Specify the layout of multi-dimensional views explicitly, e.g.
  `Kokkos::View<double**, Kokkos::LayoutLeft>`. The default layout differs
  between backends, so code that relies on it can work on a GPU but fail on the
  host, or vice versa.
- Test device code by comparing against its host counterpart. The tests run on
  the host backend in CI, which has no GPUs.
