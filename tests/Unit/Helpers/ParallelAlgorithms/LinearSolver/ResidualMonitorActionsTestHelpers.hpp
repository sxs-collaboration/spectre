// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <converse.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace ResidualMonitorActionsTestHelpers {

struct CheckObservationIdTag : db::SimpleTag {
  using type = observers::ObservationId;
};

struct CheckSubfileNameTag : db::SimpleTag {
  using type = std::string;
};

struct CheckReductionNamesTag : db::SimpleTag {
  using type = std::vector<std::string>;
};

struct CheckReductionDataTag : db::SimpleTag {
  using type = std::tuple<size_t, double>;
};

using observer_writer_tags =
    tmpl::list<CheckObservationIdTag, CheckSubfileNameTag,
               CheckReductionNamesTag, CheckReductionDataTag>;

struct MockWriteReductionData {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename... ReductionDatums,
            Requires<tmpl::list_contains_v<DbTagsList, CheckObservationIdTag>> =
                nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    std::vector<std::string>&& reduction_names,
                    Parallel::ReductionData<ReductionDatums...>&&
                        in_reduction_data) noexcept {
    db::mutate<CheckObservationIdTag, CheckSubfileNameTag,
               CheckReductionNamesTag, CheckReductionDataTag>(
        make_not_null(&box),
        [ observation_id, subfile_name, reduction_names, in_reduction_data ](
            const gsl::not_null<typename CheckObservationIdTag::type*>
                check_observation_id,
            const gsl::not_null<typename CheckSubfileNameTag::type*>
                check_subfile_name,
            const gsl::not_null<typename CheckReductionNamesTag::type*>
                check_reduction_names,
            const gsl::not_null<typename CheckReductionDataTag::type*>
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
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<observer_writer_tags>>>>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;

  using replace_these_threaded_actions =
      tmpl::list<observers::ThreadedActions::WriteReductionData>;
  using with_these_threaded_actions = tmpl::list<MockWriteReductionData>;
};

}  // namespace ResidualMonitorActionsTestHelpers
