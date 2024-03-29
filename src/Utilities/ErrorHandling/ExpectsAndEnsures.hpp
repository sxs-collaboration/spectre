// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macros Expects and Ensures

#pragma once

#include "Utilities/ErrorHandling/Error.hpp"

// part of GSL, but GSL depends on Expects and Ensures...
#ifndef UNLIKELY
#if defined(__clang__) || defined(__GNUC__)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define UNLIKELY(x) (x)
#endif
#endif

/*!
 * \ingroup ErrorHandlingGroup
 * \brief check expectation of pre-conditions of a function
 *
 * The Expects macro sets the preconditions to a function's arguments, it is a
 * contract (C++20) that must be satisfied. See the CppCoreGuidelines for
 * details.
 * \param cond the expression that is expected to be true
 */
#if defined(SPECTRE_DEBUG) || defined(EXPECTS_ENSURES)
#define Expects(cond)      \
  if (UNLIKELY(!(cond))) { \
    ERROR(#cond);          \
  } else                   \
    static_cast<void>(0)
#else
#define Expects(cond)        \
  if (false) {               \
    static_cast<void>(cond); \
  } else                     \
    static_cast<void>(0)
#endif

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Check that a post-condition of a function is true
 *
 * The Ensures macro sets the postconditions of function, it is a contract
 * (C++20) that must be satisfied. See the CppCoreGuidelines for details.
 * \param cond the expression that is expected to be true
 */
#if defined(SPECTRE_DEBUG) || defined(EXPECTS_ENSURES)
#define Ensures(cond)      \
  if (UNLIKELY(!(cond))) { \
    ERROR(#cond);          \
  } else                   \
    static_cast<void>(0)
#else
#define Ensures(cond)        \
  if (false) {               \
    static_cast<void>(cond); \
  } else                     \
    static_cast<void>(0)
#endif
