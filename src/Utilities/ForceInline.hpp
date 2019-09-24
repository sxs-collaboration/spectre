// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macro to always inline a function.

#pragma once

#if defined(__GNUC__)
/// \ingroup UtilitiesGroup
/// Always inline a function. Only use this if you benchmarked the code.
#define SPECTRE_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
/// \ingroup UtilitiesGroup
/// Always inline a function. Only use this if you benchmarked the code.
#define SPECTRE_ALWAYS_INLINE inline
#endif
