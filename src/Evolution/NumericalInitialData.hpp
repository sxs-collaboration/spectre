// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits.hpp"

/// Empty base class for marking a class as numerical initial data.
///
/// \see `is_numeric_initial_data`
struct MarkAsNumericalInitialData {};

// @{
/// Checks if the class `T` is marked as numerical initial data.
template <typename T>
using is_numerical_initial_data =
  typename std::is_convertible<T*, MarkAsNumericalInitialData*>;

template <typename T>
constexpr bool is_numerical_initial_data_v =
  cpp17::is_convertible_v<T*, MarkAsNumericalInitialData*>;
// @}

/// Provides compile-time information for importing numerical initial data for
/// the `System` from a data file.
template <typename System>
struct NumericalInitialData : MarkAsNumericalInitialData {
    using import_fields = typename System::initial_data_fields_tag;
};

namespace OptionTags {
/*!
 * \brief Holds option tags for importing numeric data as initial guess for an
 * elliptic solve.
 */
struct NumericalInitialData {
  using group = importers::OptionTags::Group;
  static constexpr OptionString help = "Initial data to read";
};
}  // namespace OptionTags
