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

# Writing code for devices

- Annotate functions that should be callable on the device with
  `KOKKOS_FUNCTION` from `Utilities/Kokkos/KokkosMacros.hpp`. The annotation
  does nothing in builds without Kokkos, so no other changes are needed. This
  header only includes the Kokkos macros, so it can be included anywhere.
- Put code that uses Kokkos types, such as `Kokkos::View` or
  `Kokkos::parallel_for`, in a separate file named after its host counterpart
  with a `Kokkos` suffix, e.g. `VariablesKokkos.hpp`, and include
  `Utilities/Kokkos/KokkosCore.hpp` there. Wrap everything after the
  includes in `#ifdef SPECTRE_KOKKOS`, which is defined when building with
  Kokkos. Add Kokkos-only tests to the build only `if(SPECTRE_KOKKOS)`. Avoid
  `#ifdef SPECTRE_KOKKOS` branches within files shared with host code.
- Specify the layout of multi-dimensional views explicitly, e.g.
  `Kokkos::View<double**, Kokkos::LayoutLeft>`. The default layout differs
  between backends, so code that relies on it can work on a GPU but fail on the
  host, or vice versa.
- Don't call `Kokkos::fence()` after launching kernels. Kernels launched on the
  same execution space instance run in order, and copies to the host such as
  `copy_to_host` or `Kokkos::deep_copy` wait for earlier kernels to finish.
  Fence only before host code reads memory written by a kernel without such a
  copy (e.g. a host-accessible view), before measuring time, or to synchronize
  several execution space instances. In the latter case fence only the instance
  that needs it, e.g. `exec.fence()`.
- Test device code by comparing against its host counterpart. The tests run on
  the host backend in CI, which has no GPUs.

## The C++ standard library in device code

\warning Don't call standard library algorithms in device code, i.e. in
`KOKKOS_FUNCTION`s and Kokkos kernels. This includes `<algorithm>`,
`<numeric>`, `std::ranges` and `std::views`, the comparison operators of
`std::array` (`==`, `<`, ...), and `std::complex`.

CUDA doesn't support the host standard library (libstdc++) in device code.
SpECTRE compiles device code with `--expt-relaxed-constexpr`, which lets device
code call `constexpr` host functions, so many standard library functions compile
for the device anyway. Some of them then give wrong results without any error
or warning. In tests on an NVIDIA GH200, for example:

- `std::array`'s `operator==`, `std::equal`, `std::copy` and `std::complex`
  wrote no results at all: their kernels contained no instructions to write the
  results.
- `std::sort` (for some inputs), `std::rotate`, `std::upper_bound` and
  `std::lcm` gave wrong values.
- The `std::ranges` algorithms didn't compile for the device.

One reason is that libstdc++ uses GCC builtins that CUDA doesn't support in
device code, such as `__builtin_memmove`, `__builtin_memcmp` and
`__builtin_mul_overflow`. Called directly in device code they are a compile
error, but inside standard library functions they are compiled without an error
and give wrong device code. Whether a function is affected depends on how
libstdc++ implements it, so a function that works in a test isn't guaranteed to
work for other types, inputs or compiler versions.

These parts of the standard library gave correct results in device code in
those tests and can be used:

- `<cmath>` functions on `float` and `double`, which CUDA implements for the
  device. Their results can differ from the host in the last bits, and
  floating-point errors such as `std::sqrt(-1.0)` aren't trapped on the device.
- `std::numeric_limits`
- `std::array` element access (with `gsl::at`), iterators and range-based for
  loops
- `std::min`, `std::max`, `std::clamp` and `std::swap`
- `std::pair`, `std::tuple`, `std::optional` and `std::span` (without
  algorithms)

Type traits, `std::move` and `std::forward` can be used too, since they don't
generate any code of their own.

Instead of standard library algorithms, use the `alg` functions from
`Utilities/Algorithm.hpp`. For containers with contiguous iterators, such as
`std::array`, they are implemented as plain loops and can
be called in device code (except `alg::sample` and `alg::sort`). For other
containers they can only be called on the host. `alg::sort` aborts on
the device; sort short lists with `alg::bubble_sort` instead, which takes
\f$O(N^2)\f$ operations. For algorithms over data on the device, such as
reductions over all grid points, use Kokkos (e.g. `Kokkos::parallel_reduce`)
instead.

A possible alternative is CUDA's own standard library,
[libcu++](https://nvidia.github.io/cccl/unstable/libcudacxx/index.html)
(`cuda::std`), which is written for device code. In the same tests on an NVIDIA
GH200, `cuda::std` versions of the functions above gave correct
results. However, libcu++ is only available with CUDA, not with the other Kokkos
GPU backends (HIP and SYCL), and its version shipped with CUDA 13 doesn't
provide `cuda::std::sort`. SpECTRE does not currently use it.

## Enforcement and testing

- The `device_std_algorithms` check in `tools/FileTestDefs.sh`, which runs in
  CI, rejects calls to standard library algorithms in files under `src/` that
  contain device code. Mark host-only code in such files with
  `// NOLINT(spectre-device-std)`. The check is textual, so it can't find
  standard library calls in functions that device code calls in other files,
  or uses of `std::array`'s comparison operators.
- nvcc warns about calls to host functions from device code ("calling a
  `__host__` function from a `__host__ __device__` function"). Treat these
  warnings as errors in the code you write.
- Code that compiles for the device can still give wrong results on the
  device. Every function that can be called in device code needs a test that
  calls it in a Kokkos kernel, e.g. with `TestHelpers::check_on_device`, and
  these tests must be run on a GPU before code is merged. CI doesn't run tests
  on GPUs.
