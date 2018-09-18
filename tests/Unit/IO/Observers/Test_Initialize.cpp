// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>
// IWYU pragma: no_include <exception>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/ArrayComponentId.hpp"  // IWYU pragma: keep
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
template <typename Metavariables>
struct observer_component
    : ActionTesting::MockArrayComponent<Metavariables, size_t, tmpl::list<>,
                                        tmpl::list<>> {
  using initial_databox = db::compute_databox_type<
      typename observers::Actions::Initialize::return_tag_list>;
};

struct Metavariables {
  using component_list = tmpl::list<observer_component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE("Unit.IO.Observers.Initialize", "[Unit][Observers]") {
  using LocalAlgorithms =
      typename ActionTesting::MockRuntimeSystem<Metavariables>::LocalAlgorithms;
  using obs_component = observer_component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using ObserverLocalAlgsTag =
      typename MockRuntimeSystem::template LocalAlgorithmsTag<obs_component>;
  LocalAlgorithms local_algs{};
  tuples::get<ObserverLocalAlgsTag>(local_algs)
      .emplace(0, ActionTesting::MockLocalAlgorithm<obs_component>{});

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{},
                                                         std::move(local_algs)};

  runner.simple_action<obs_component, observers::Actions::Initialize>(0);
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
