// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/ArrayComponentId.hpp"  // IWYU pragma: keep
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <exception>

namespace {
template <typename Metavariables>
struct observer_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename observers::Actions::Initialize<Metavariables>::return_tag_list>;
};

struct Metavariables {
  using component_list = tmpl::list<observer_component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using observed_reduction_data_tags = tmpl::list<>;

  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE("Unit.IO.Observers.Initialize", "[Unit][Observers]") {
  using TupleOfMockDistributedObjects =
      typename ActionTesting::MockRuntimeSystem<
          Metavariables>::TupleOfMockDistributedObjects;
  using obs_component = observer_component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using ObserverMockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          obs_component>;
  TupleOfMockDistributedObjects dist_objects{};
  tuples::get<ObserverMockDistributedObjectsTag>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<obs_component>{});

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {}, std::move(dist_objects)};

  runner.simple_action<obs_component,
                       observers::Actions::Initialize<Metavariables>>(0);
  const auto& observer_box =
      runner.template algorithms<obs_component>()
          .at(0)
          .template get_databox<typename obs_component::initial_databox>();
  CHECK(db::get<observers::Tags::NumberOfEvents>(observer_box).empty());
  CHECK(db::get<observers::Tags::ReductionArrayComponentIds>(observer_box)
            .empty());
  CHECK(
      db::get<observers::Tags::VolumeArrayComponentIds>(observer_box).empty());
  CHECK(db::get<observers::Tags::TensorData>(observer_box).empty());
}
}  // namespace
