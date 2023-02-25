// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeStepId;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct TimeStepId;
}  // namespace Tags
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace imex::Actions {
/// \ingroup ActionsGroup
/// \brief Records the implicit sources in the implicit time stepper history.
///
/// Uses:
/// - GlobalCache: nothing
/// - DataBox:
///   - Tags::TimeStepId
///   - system::variables_tag
///   - as required by source terms
///
/// DataBox changes:
/// - imex::Tags::ImplicitHistory<sector> for each sector
template <typename System>
struct RecordTimeStepperData {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT const
    static_assert(tt::assert_conforms_to_v<System, protocols::ImexSystem>);

    const size_t number_of_grid_points =
        db::get<typename System::variables_tag>(box).number_of_grid_points();

    tmpl::for_each<typename System::implicit_sectors>([&](auto sector_v) {
      using sector = tmpl::type_from<decltype(sector_v)>;
      using source =
          typename tmpl::front<typename sector::solve_attempts>::source;
      using history_tag = Tags::ImplicitHistory<sector>;
      using DtSectorVars =
          Variables<db::wrap_tags_in<::Tags::dt, typename sector::tensors>>;
      db::mutate_apply<
          tmpl::list<history_tag>,
          tmpl::push_front<typename source::argument_tags, ::Tags::TimeStepId>>(
          [&](const gsl::not_null<typename history_tag::type*> history,
              const TimeStepId& time_step_id, const auto&... source_arguments) {
            history->insert_in_place(
                time_step_id, history_tag::type::no_value,
                [&](const gsl::not_null<DtSectorVars*> source_result) {
                  source_result->initialize(number_of_grid_points);
                  tmpl::as_pack<typename source::return_tags>(
                      [&](auto... source_tags) {
                        // The history stores derivatives as
                        // ::Tags::dt<Var>, but the source provides
                        // ::Tags::Source<Var>.  Since the only
                        // implicit equations we support are source
                        // terms, these quantities are equal, but we
                        // still need to make the types match.
                        source::apply(
                            make_not_null(
                                &get<::Tags::dt<db::remove_tag_prefix<
                                    tmpl::type_from<decltype(source_tags)>>>>(
                                    *source_result))...,
                            source_arguments...);
                      });
                });
          },
          make_not_null(&box));
    });

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace imex::Actions
