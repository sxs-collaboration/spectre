// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <converse.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel

namespace ResidualMonitorActionsTestHelpers {

struct CheckObservationIdTag : db::SimpleTag {
  using type = observers::ObservationId;
  static std::string name() noexcept { return "CheckObservationIdTag"; }
};

struct CheckSubfileNameTag : db::SimpleTag {
  using type = std::string;
  static std::string name() noexcept { return "CheckSubfileNameTag"; }
};

struct CheckReductionNamesTag : db::SimpleTag {
  using type = std::vector<std::string>;
  static std::string name() noexcept { return "CheckReductionNamesTag"; }
};

struct CheckReductionDataTag : db::SimpleTag {
  using type = std::tuple<size_t, double>;
  static std::string name() noexcept { return "CheckReductionDataTag"; }
};

using observer_writer_tags =
    tmpl::list<CheckObservationIdTag, CheckSubfileNameTag,
               CheckReductionNamesTag, CheckReductionDataTag>;

struct MockWriteReductionData {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex,
            typename... ReductionDatums>
  static void apply(db::DataBox<observer_writer_tags>& box,  // NOLINT
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const gsl::not_null<CmiNodeLock*> /*node_lock*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    std::vector<std::string>&& reduction_names,
                    Parallel::ReductionData<ReductionDatums...>&&
                        in_reduction_data) noexcept {
    db::mutate<CheckObservationIdTag, CheckSubfileNameTag,
               CheckReductionNamesTag, CheckReductionDataTag>(
        make_not_null(&box),
        [ observation_id, subfile_name, reduction_names, in_reduction_data ](
            const gsl::not_null<db::item_type<CheckObservationIdTag>*>
                check_observation_id,
            const gsl::not_null<db::item_type<CheckSubfileNameTag>*>
                check_subfile_name,
            const gsl::not_null<db::item_type<CheckReductionNamesTag>*>
                check_reduction_names,
            const gsl::not_null<db::item_type<CheckReductionDataTag>*>
                check_reduction_data) noexcept {
          *check_observation_id = observation_id;
          *check_subfile_name = subfile_name;
          *check_reduction_names = reduction_names;
          *check_reduction_data = in_reduction_data.data();
        });
  }
};

template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using initial_databox = db::compute_databox_type<observer_writer_tags>;

  using replace_these_threaded_actions =
      tmpl::list<observers::ThreadedActions::WriteReductionData>;
  using with_these_threaded_actions = tmpl::list<MockWriteReductionData>;
};

}  // namespace ResidualMonitorActionsTestHelpers
