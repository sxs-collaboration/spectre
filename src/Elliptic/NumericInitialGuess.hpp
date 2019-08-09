// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits.hpp"

namespace elliptic {

/// Empty base class for marking a class as a numeric initial guess.
///
/// \see `elliptic::is_numeric_initial_guess`
struct MarkAsNumericInitialGuess {};

// @{
/// Checks if the class `T` is marked as a numeric initial guess.
template <typename T>
using is_numeric_initial_guess =
    typename std::is_convertible<T*, MarkAsNumericInitialGuess*>;

template <typename T>
constexpr bool is_numeric_initial_guess_v =
    cpp17::is_convertible_v<T*, MarkAsNumericInitialGuess*>;
// @}

/// Provides compile-time information for importing a numeric initial guess for
/// the `System` from a data file.
template <typename System>
struct NumericInitialGuess : MarkAsNumericInitialGuess {
  using import_fields =
      db::get_variables_tags_list<typename System::fields_tag>;
};

}  // namespace elliptic
