// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Evolution/Imex/SolveImplicitSector.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/CleanupRoutine.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
template <typename Tag>
struct Next;
struct Time;
struct TimeStepId;
}  // namespace Tags
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace imex::Actions {
/// \ingroup ActionsGroup
/// \brief Perform implicit variable updates for one substep
///
/// Uses:
/// - DataBox:
///   - Tags::Next<Tags::TimeStepId>
///   - Tags::Time
///   - Tags::TimeStep
///   - Tags::TimeStepper<ImexTimeStepper>
///   - imex::Tags::Mode
///   - imex::Tags::SolveTolerance
///   - as required by system implicit sectors
///
/// DataBox changes:
/// - variables_tag
/// - imex::Tags::ImplicitHistory<sector> for each sector
template <typename System>
struct DoImplicitStep {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(tt::assert_conforms_to_v<System, protocols::ImexSystem>);

    const double original_time = db::get<::Tags::Time>(box);
    const CleanupRoutine reset_time = [&]() {
      db::mutate<::Tags::Time>(
          [&](const gsl::not_null<double*> time) { *time = original_time; },
          make_not_null(&box));
    };
    db::mutate<::Tags::Time>(
        [](const gsl::not_null<double*> time,
           const TimeStepId& next_time_step_id) {
          *time = next_time_step_id.substep_time();
        },
        make_not_null(&box), db::get<::Tags::Next<::Tags::TimeStepId>>(box));

    tmpl::for_each<typename System::implicit_sectors>([&](auto sector_v) {
      using sector = tmpl::type_from<decltype(sector_v)>;
      db::mutate_apply<
          SolveImplicitSector<typename System::variables_tag, sector>>(
          make_not_null(&box));
    });
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace imex::Actions
