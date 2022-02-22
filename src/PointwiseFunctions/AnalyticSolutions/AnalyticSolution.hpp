// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

// note that std::is_convertible is used in the following type aliases as it
// will not match private base classes (unlike std::is_base_of) and some
// analytic data privately inherits from an analytic solution

/// \ingroup AnalyticSolutionsGroup
/// \brief Empty base class for marking analytic solutions.
struct MarkAsAnalyticSolution {};

/// \ingroup AnalyticSolutionsGroup
/// \brief Check if `T` is an analytic solution
template <typename T>
using is_analytic_solution =
    typename std::is_convertible<T*, MarkAsAnalyticSolution*>;

/// \ingroup AnalyticSolutionsGroup
/// \brief `true` if `T` is an analytic solution
template <typename T>
constexpr bool is_analytic_solution_v =
    std::is_convertible_v<T*, MarkAsAnalyticSolution*>;
