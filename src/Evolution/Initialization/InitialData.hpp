// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "Evolution/TypeTraits.hpp"  // IWYU pragma: keep
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/FakeVirtual.hpp"

namespace evolution {
/// Extract initial data either from analytic data or from an analytic
/// solution at a specified time.
template <typename SolutionOrData, typename Coordinates, typename... Tags,
          Requires<evolution::is_analytic_solution_v<SolutionOrData>> = nullptr>
decltype(auto) initial_data(const SolutionOrData& solution_or_data,
                            Coordinates&& coordinates, const double time,
                            const tmpl::list<Tags...> tags) {
  return solution_or_data.variables(std::forward<Coordinates>(coordinates),
                                    time, tags);
}

/// \cond
template <typename SolutionOrData, typename Coordinates, typename... Tags,
          Requires<evolution::is_analytic_data_v<SolutionOrData>> = nullptr>
decltype(auto) initial_data(const SolutionOrData& solution_or_data,
                            Coordinates&& coordinates, const double /*time*/,
                            const tmpl::list<Tags...> tags) {
  return solution_or_data.variables(std::forward<Coordinates>(coordinates),
                                    tags);
}
/// \endcond

template <typename DerivedClasses, typename InitialData, typename Coordinates,
          typename... Tags>
tuples::TaggedTuple<Tags...> initial_data(
    const InitialData& initial_data, Coordinates&& coordinates,
    const double time, const tmpl::list<Tags...> /*tags*/) noexcept {
  return call_with_dynamic_type<tuples::TaggedTuple<Tags...>, DerivedClasses>(
      &initial_data,
      [&coordinates, &time](auto* const initial_data_derived) noexcept {
        using derived = std::decay_t<decltype(*initial_data_derived)>;

        static_assert(evolution::is_analytic_data_v<derived> xor
                          evolution::is_analytic_solution_v<derived>,
                      "initial_data must be either an analytic_data or an "
                      "analytic_solution");

        if constexpr (evolution::is_analytic_data_v<derived>) {
          return initial_data_derived->variables(
              std::forward<Coordinates>(coordinates), tmpl::list<Tags...>{});
        } else if constexpr (evolution::is_analytic_solution_v<derived>) {
          return initial_data_derived->variables(
              std::forward<Coordinates>(coordinates), time,
              tmpl::list<Tags...>{});
        }
      });
}
}  // namespace evolution
