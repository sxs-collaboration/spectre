// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Macros for code that can run on GPUs. This header only includes the light
/// Kokkos headers that define them, so it can be included everywhere. Include
/// `Utilities/Kokkos/KokkosCore.hpp` instead to use Kokkos types and parallel
/// loops such as `Kokkos::View` and `Kokkos::parallel_for`.

#pragma once

// The build system defines `SPECTRE_KOKKOS` when configured with
// `-D SPECTRE_KOKKOS=ON`.
#ifdef SPECTRE_KOKKOS
// `KOKKOS_FUNCTION`, `KOKKOS_IF_ON_HOST`, `KOKKOS_IF_ON_DEVICE`, ...
#include <Kokkos_Macros.hpp>
// `Kokkos::abort`
#include <Kokkos_Abort.hpp>

/*!
 * \brief Defined while compiling device code for a GPU backend.
 *
 * GPU compilers compile each source file twice: once for the host and once for
 * the device. This macro is defined only in the device pass, so code that is
 * never called on the device can be excluded from device compilation. For
 * explicit instantiations use `SPECTRE_DEVICE_AND_HOST_TYPES` instead.
 *
 * Only use this for code at namespace scope. Inside functions use Kokkos'
 * `KOKKOS_IF_ON_HOST` and `KOKKOS_IF_ON_DEVICE` instead.
 */
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__) or \
    defined(__SYCL_DEVICE_ONLY__)
#define SPECTRE_KOKKOS_DEVICE_PASS 1
#endif

#else  // SPECTRE_KOKKOS
#define KOKKOS_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
// Without Kokkos all code runs on the host. The argument is wrapped in
// parentheses, e.g. `KOKKOS_IF_ON_HOST((io_call(...);))`, like in Kokkos.
#define SPECTRE_KOKKOS_STRIP_PARENS(...) __VA_ARGS__
#define KOKKOS_IF_ON_HOST(CODE) {SPECTRE_KOKKOS_STRIP_PARENS CODE}
#define KOKKOS_IF_ON_DEVICE(CODE) \
  {                               \
  }
#endif  // SPECTRE_KOKKOS

/*!
 * \brief Put before a `KOKKOS_FUNCTION` template whose instantiations for
 * some types can only run on the host.
 *
 * Examples are templates that call a callable passed by the caller (e.g. the
 * function of a root finder), or header-defined templates that are also
 * instantiated for `DataVector`. nvcc warns about these host-only
 * instantiations, even if they are only used on the host. This disables that
 * check for the function that follows, for all its instantiations.
 *
 * The compiler then also doesn't diagnose device code that uses a host-only
 * instantiation. See \ref gpu_support.
 *
 * Define `SPECTRE_KOKKOS_ENABLE_ALL_EXEC_CHECKS` (e.g. in `CMAKE_CXX_FLAGS`) to
 * make this macro do nothing. nvcc then warns about all host-only
 * instantiations, which is useful to check the device-capable instantiations
 * (e.g. by searching the warnings for `T=double`).
 */
#if defined(__NVCC__) and not defined(SPECTRE_KOKKOS_ENABLE_ALL_EXEC_CHECKS)
#define SPECTRE_KOKKOS_DISABLE_EXEC_CHECK _Pragma("nv_exec_check_disable")
#else
#define SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
#endif

/*!
 * \brief The types to instantiate a function for with `GENERATE_INSTANTIATIONS`
 * when only some of them are used on GPUs.
 *
 * Functions marked `KOKKOS_FUNCTION` can run on the CPU and, in GPU builds, on
 * a GPU. They are instantiated for the `DEVICE_TYPES` on the GPU and for the
 * `HOST_TYPES` on the CPU. List types that only exist on the CPU, such as
 * `DataVector`, only in `HOST_TYPES`. The `HOST_TYPES` should include the
 * `DEVICE_TYPES`, so the function can be called for them on the CPU as well.
 * For example,
 *
 * \code
 * GENERATE_INSTANTIATIONS(
 *     INSTANTIATE, (1, 2, 3),
 *     SPECTRE_DEVICE_AND_HOST_TYPES((double), (double, DataVector)))
 * \endcode
 *
 * instantiates the function for `double` on the GPU, and for `double` and
 * `DataVector` on the CPU. In builds without a GPU, this is simply the list of
 * `HOST_TYPES`.
 */
#ifdef SPECTRE_KOKKOS_DEVICE_PASS
#define SPECTRE_DEVICE_AND_HOST_TYPES(DEVICE_TYPES, HOST_TYPES) DEVICE_TYPES
#else
#define SPECTRE_DEVICE_AND_HOST_TYPES(DEVICE_TYPES, HOST_TYPES) HOST_TYPES
#endif
