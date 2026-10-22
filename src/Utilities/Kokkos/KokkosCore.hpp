// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Includes Kokkos for code that uses Kokkos types and parallel loops, such as
/// `Kokkos::View` and `Kokkos::parallel_for`. Code that only needs the macros
/// to annotate functions (e.g. `KOKKOS_FUNCTION`) should include
/// `Utilities/Kokkos/KokkosMacros.hpp` instead.

#pragma once

#include "Utilities/Kokkos/KokkosMacros.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Core.hpp>
#endif  // SPECTRE_KOKKOS
