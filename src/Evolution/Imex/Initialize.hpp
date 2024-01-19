// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Evolution/Imex/Tags/Mode.hpp"
#include "Evolution/Imex/Tags/SolveFailures.hpp"
#include "Evolution/Imex/Tags/SolveTolerance.hpp"
#include "Time/History.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace imex {
/// Create the IMEX structures and options.
template <typename System, typename = typename System::implicit_sectors>
struct Initialize;

/// \cond
template <typename System, typename... Sectors>
struct Initialize<System, tmpl::list<Sectors...>> {
  static_assert(tt::assert_conforms_to_v<System, protocols::ImexSystem>);

  using example_tensor_tag =
      tmpl::front<typename tmpl::front<tmpl::list<Sectors...>>::tensors>;

  using const_global_cache_tags = tmpl::list<Tags::Mode, Tags::SolveTolerance>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags = tmpl::list<Tags::ImplicitHistory<Sectors>...,
                                 Tags::SolveFailures<Sectors>...>;
  using compute_tags = tmpl::list<>;

  using return_tags = simple_tags;
  using argument_tags =
      tmpl::list<::Tags::HistoryEvolvedVariables<>, example_tensor_tag>;

  static void apply(
      const gsl::not_null<
          typename Tags::ImplicitHistory<Sectors>::type*>... histories,
      const gsl::not_null<
          typename Tags::SolveFailures<Sectors>::type*>... solve_failures,
      const TimeSteppers::History<typename System::variables_tag::type>&
          explicit_history,
      const typename example_tensor_tag::type& example_tensor) {
    const auto order = explicit_history.integration_order();
    expand_pack((histories->integration_order(order), 0)...);
    expand_pack(*solve_failures = make_with_value<Scalar<DataVector>>(
                    example_tensor, 0.0)...);
  }
};
/// \endcond
}  // namespace imex
