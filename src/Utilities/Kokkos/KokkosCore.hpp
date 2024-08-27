// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#if __has_include(<Kokkos_Core.hpp>)
#include <Kokkos_Core.hpp>

/// \brief If defined then SpECTRE is using Kokkos
#define SPECTRE_KOKKOS 1

#else  // #if __has_include(<Kokkos_Core.hpp>)
#define KOKKOS_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#endif  // #if __has_include(<Kokkos_Core.hpp>)
