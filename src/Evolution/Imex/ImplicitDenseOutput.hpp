// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace Tags {
struct Time;
}  // namespace Tags
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace imex::Tags {
template <typename ImplicitSector>
struct ImplicitHistory;
}  // namespace imex::Tags
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace imex {
/// Mutator to apply the implicit portion of dense output, intended
/// for use in `RunEventsAndDenseTriggers`.
template <typename ImexSystem>
struct ImplicitDenseOutput {
  static_assert(tt::assert_conforms_to_v<ImexSystem, protocols::ImexSystem>);

  using return_tags = tmpl::list<typename ImexSystem::variables_tag>;
  using argument_tags = tmpl::push_front<
      tmpl::transform<typename ImexSystem::implicit_sectors,
                      tmpl::bind<Tags::ImplicitHistory, tmpl::_1>>,
      ::Tags::TimeStepper<TimeStepper>, ::Tags::Time>;

  template <typename... Histories>
  static void apply(
      const gsl::not_null<typename ImexSystem::variables_tag::type*> variables,
      const TimeStepper& time_stepper, const double time,
      const Histories&... histories) {
    const auto update_sector = [&](auto sector_v, auto history) {
      using sector = tmpl::type_from<decltype(sector_v)>;
      auto sector_variables =
          variables->template reference_subset<typename sector::tensors>();
      const bool success = time_stepper.dense_update_u(
          make_not_null(&sector_variables), history, time);
      if (not success) {
        ERROR("IMEX dense output failed with history:\n" << history);
      }
    };
    tmpl::as_pack<typename ImexSystem::implicit_sectors>(
        [&](auto... sectors_v) {
          EXPAND_PACK_LEFT_TO_RIGHT(update_sector(sectors_v, histories));
        });
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ParallelComponent>
  static bool is_ready(
      const gsl::not_null<db::DataBox<DbTagsList>*> /*box*/,
      const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* /*component*/) {
    return true;
  }
};
}  // namespace imex
