// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

// note that std::is_convertible is used in the following type aliases as it
// will not match private base classes (unlike std::is_base_of) and some
// analytic data privately inherits from an analytic solution

/// \ingroup AnalyticDataGroup
/// \brief Empty base class for marking analytic data.
struct MarkAsAnalyticData {};

/// \ingroup AnalyticDataGroup
/// \brief Check if `T` is an analytic data
template <typename T>
using is_analytic_data = typename std::is_convertible<T*, MarkAsAnalyticData*>;

/// \ingroup AnalyticDataGroup
/// \brief `true` if `T` is an analytic data
template <typename T>
constexpr bool is_analytic_data_v =
    std::is_convertible_v<T*, MarkAsAnalyticData*>;
