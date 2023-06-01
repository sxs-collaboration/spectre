// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "Evolution/TypeTraits.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::Initialization {
/// Extract initial data either from analytic data or from an analytic
/// solution at a specified time.
template <typename SolutionOrData, typename Coordinates, typename... Tags,
          Requires<is_analytic_solution_v<SolutionOrData>> = nullptr>
decltype(auto) initial_data(const SolutionOrData& solution_or_data,
                            Coordinates&& coordinates, const double time,
                            const tmpl::list<Tags...> tags) {
  return solution_or_data.variables(std::forward<Coordinates>(coordinates),
                                    time, tags);
}

/// \cond
template <typename SolutionOrData, typename Coordinates, typename... Tags,
          Requires<is_analytic_data_v<SolutionOrData> or
                   is_numeric_initial_data_v<SolutionOrData>> = nullptr>
decltype(auto) initial_data(const SolutionOrData& solution_or_data,
                            Coordinates&& coordinates, const double /*time*/,
                            const tmpl::list<Tags...> tags) {
  return solution_or_data.variables(std::forward<Coordinates>(coordinates),
                                    tags);
}
/// \endcond
}  // namespace evolution::Initialization
