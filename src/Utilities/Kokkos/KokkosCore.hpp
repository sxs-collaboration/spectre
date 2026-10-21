// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// The build system defines `SPECTRE_KOKKOS` when configured with
// `-D SPECTRE_KOKKOS=ON`.
#ifdef SPECTRE_KOKKOS
#include <Kokkos_Core.hpp>
#else  // SPECTRE_KOKKOS
#define KOKKOS_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#endif  // SPECTRE_KOKKOS
