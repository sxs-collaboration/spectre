// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Commonly used routines, functions and definitions shared amongst unit tests

#pragma once

#include <catch.hpp>
#include <charm++.h>
#include <pup.h>

#include "Parallel/Abort.hpp"
#include "Parallel/Serialize.hpp"

/*!
 * \ingroup TestingFramework
 * \brief Mark a test to be checking an ASSERT
 *
 * \details
 * Testing error handling is just as important as testing functionality. Tests
 * that are supposed to exit with an error must be annotated with the attribute
 * \code
 * // [[OutputRegex, The regex that should be found in the output]]
 * \endcode
 * Note that the regex only needs to be a sub-expression of the error message,
 * that is, there are implicit wildcards before and after the string.
 *
 * In order to test ASSERT's properly the test must also fail for release
 * builds. This is done by adding this macro at the beginning for the test.
 *
 * \example
 * snippet Test_Time.cpp example_of_error_test
 */
#ifdef SPECTRE_DEBUG
#define ASSERTION_TEST() \
  do {                   \
  } while (false)
#else
#include "Parallel/Abort.hpp"
#define ASSERTION_TEST() \
  Parallel::abort("### No ASSERT tests in release mode ###")
#endif

/*!
 * \ingroup TestingFramework
 * \brief Serializes and deserializes an object `t` of type `T`
 *
 * \example
 * To test serialization of copyable types use:
 *
 * snippet Test_PupStlCpp11.cpp example_serialize_copyable
 *
 * and for non-copyable types use:
 * snippet Test_PupStlCpp11.cpp example_serialize_non_copyable
 */
template <typename T>
T serialize_and_deserialize(const T& t) {
  return deserialize<T>(serialize<T>(t).data());
}
