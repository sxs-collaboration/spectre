// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Evolution/Protocols.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace evolution {
// note that std::is_convertible is used in the following type aliases as it
// will not match private base classes (unlike std::is_base_of) and some
// analytic data privately inherits from an analytic solution

/// \ingroup AnalyticDataGroup
template <typename T>
using is_analytic_data = typename std::is_convertible<T*, MarkAsAnalyticData*>;

/// \ingroup AnalyticDataGroup
template <typename T>
constexpr bool is_analytic_data_v =
    std::is_convertible_v<T*, MarkAsAnalyticData*>;

/// \ingroup AnalyticSolutionsGroup
template <typename T>
using is_analytic_solution =
    typename std::is_convertible<T*, MarkAsAnalyticSolution*>;

/// \ingroup AnalyticSolutionsGroup
template <typename T>
constexpr bool is_analytic_solution_v =
    std::is_convertible_v<T*, MarkAsAnalyticSolution*>;

/// @{
/// Helper metafunction that checks if the class `T` is marked as numeric
/// initial data.
template <typename T>
using is_numeric_initial_data =
    tt::conforms_to<T, protocols::NumericInitialData>;

template <typename T>
constexpr bool is_numeric_initial_data_v =
    tt::conforms_to_v<T, protocols::NumericInitialData>;
/// @}
}  // namespace evolution
