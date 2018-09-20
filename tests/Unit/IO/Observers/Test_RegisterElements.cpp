// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <exception>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"               // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
using ElementIndexType = ElementIndex<2>;

template <typename Metavariables>
struct element_component
    : ActionTesting::MockArrayComponent<Metavariables, ElementIndexType,
                                        tmpl::list<>, tmpl::list<>> {
  using initial_databox = db::DataBox<tmpl::list<>>;
};

template <typename Metavariables>
struct observer_component
    : ActionTesting::MockArrayComponent<Metavariables, size_t, tmpl::list<>,
                                        tmpl::list<>,
                                        observers::Observer<Metavariables>> {
  using simple_tags = observers::Actions::Initialize::simple_tags;
  using compute_tags = observers::Actions::Initialize::compute_tags;
  using initial_databox =
      db::compute_databox_type<tmpl::append<simple_tags, compute_tags>>;
};

struct Metavariables {
  using component_list = tmpl::list<element_component<Metavariables>,
                                    observer_component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  enum class Phase { Initialize, Exit };
};

template <observers::TypeOfObservation TypeOfObservation>
void check_observer_registration() {
  using LocalAlgorithms =
      typename ActionTesting::ActionRunner<Metavariables>::LocalAlgorithms;
  using obs_component = observer_component<Metavariables>;
  using element_comp = element_component<Metavariables>;

  using ActionRunner = ActionTesting::ActionRunner<Metavariables>;
  using ObserverLocalAlgsTag =
      typename ActionRunner::template LocalAlgorithmsTag<obs_component>;
  using ElementLocalAlgsTag =
      typename ActionRunner::template LocalAlgorithmsTag<element_comp>;
  LocalAlgorithms local_algs{};
  tuples::get<ObserverLocalAlgsTag>(local_algs)
      .emplace(0, ActionTesting::MockLocalAlgorithm<obs_component>{});

  // Specific IDs have no significance, just need different IDs.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 1}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 3}}}},
                                              {1, {{{1, 0}, {5, 4}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};
  for (const auto& id : element_ids) {
    tuples::get<ElementLocalAlgsTag>(local_algs)
        .emplace(ElementIndex<2>{id},
                 ActionTesting::MockLocalAlgorithm<element_comp>{});
  }

  ActionTesting::ActionRunner<Metavariables> runner{{}, std::move(local_algs)};

  runner.simple_action<obs_component, observers::Actions::Initialize>(0);
  // Test initial state
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

  // Register elements
  for (const auto& id : element_ids) {
    runner.simple_action<
        element_comp,
        observers::Actions::RegisterWithObservers<TypeOfObservation>>(id, 0);
  }

  // Test registration occurred as expected
  CHECK(db::get<observers::Tags::NumberOfEvents>(observer_box).empty());
  CHECK(db::get<observers::Tags::ReductionArrayComponentIds>(observer_box)
            .size() ==
        (TypeOfObservation == observers::TypeOfObservation::Volume
             ? size_t{0}
             : element_ids.size()));
  CHECK(
      db::get<observers::Tags::VolumeArrayComponentIds>(observer_box).size() ==
      (TypeOfObservation == observers::TypeOfObservation::Reduction
           ? size_t{0}
           : element_ids.size()));
  CHECK(db::get<observers::Tags::TensorData>(observer_box).empty());
  for (const auto& id : element_ids) {
    CHECK(
        db::get<observers::Tags::ReductionArrayComponentIds>(observer_box)
            .count(observers::ArrayComponentId(
                std::add_pointer_t<element_comp>{nullptr},
                Parallel::ArrayIndex<ElementIndex<2>>(ElementIndex<2>(id)))) ==
        (TypeOfObservation == observers::TypeOfObservation::Volume ? 0 : 1));
    CHECK(
        db::get<observers::Tags::VolumeArrayComponentIds>(observer_box)
            .count(observers::ArrayComponentId(
                std::add_pointer_t<element_comp>{nullptr},
                Parallel::ArrayIndex<ElementIndex<2>>(ElementIndex<2>(id)))) ==
        (TypeOfObservation == observers::TypeOfObservation::Reduction ? 0 : 1));
  }
}

SPECTRE_TEST_CASE("Unit.IO.Observers.RegisterElements", "[Unit]") {
  SECTION("Register as requiring reduction observer support") {
    check_observer_registration<observers::TypeOfObservation::Reduction>();
  }
  SECTION("Register as requiring volume observer support") {
    check_observer_registration<observers::TypeOfObservation::Volume>();
  }
  SECTION("Register as requiring both reduction and volume  observer support") {
    check_observer_registration<
        observers::TypeOfObservation::ReductionAndVolume>();
  }
}
}  // namespace
