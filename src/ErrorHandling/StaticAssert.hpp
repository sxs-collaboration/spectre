// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macro DEBUG_STATIC_ASSERT.

#pragma once

/*!
 * \ingroup ErrorHandlingGroup
 * \brief A `static_assert` that is only checked in Debug builds
 */
#ifdef SPECTRE_DEBUG
#define DEBUG_STATIC_ASSERT(...) static_assert(__VA_ARGS__)
#else   // ifdef  SPECTRE_DEBUG
#define DEBUG_STATIC_ASSERT(...)
#endif  // ifdef  SPECTRE_DEBUG
