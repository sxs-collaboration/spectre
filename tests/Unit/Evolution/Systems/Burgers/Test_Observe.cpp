// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/Systems/Burgers/Observe.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Linear.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <memory>
// IWYU pragma: no_include <pup.h>

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
namespace observers {
namespace Actions {
struct ContributeReductionData;
struct ContributeVolumeData;
}  // namespace Actions
}  // namespace observers

namespace {

struct MockContributeVolumeData {
  struct Results {
    observers::ObservationId observation_id{};
    std::string subfile_name{};
    observers::ArrayComponentId array_component_id{};
    std::vector<TensorComponent> in_received_tensor_data{};
    Index<1> received_extents{};
  };
  static Results results;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, size_t Dim>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    const observers::ArrayComponentId& array_component_id,
                    std::vector<TensorComponent>&& in_received_tensor_data,
                    const Index<Dim>& received_extents) noexcept {
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.array_component_id = array_component_id;
    results.in_received_tensor_data = in_received_tensor_data;
    results.received_extents = received_extents;
  }
};

MockContributeVolumeData::Results MockContributeVolumeData::results{};

struct MockContributeReductionData {
  struct Results {
    observers::ObservationId observation_id;
    std::string subfile_name;
    std::vector<std::string> reduction_names;
    double time;
    size_t number_of_grid_points;
    double u_error;
  };
  static Results results;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, typename... Ts>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    const std::vector<std::string>& reduction_names,
                    Parallel::ReductionData<Ts...>&& reduction_data) noexcept {
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.reduction_names = reduction_names;
    CHECK(reduction_data.pack_size() == 3);
    results.time = std::get<0>(reduction_data.data());
    results.number_of_grid_points = std::get<1>(reduction_data.data());
    results.u_error = std::get<2>(reduction_data.data());
  }
};

MockContributeReductionData::Results MockContributeReductionData::results{};

struct Metavariables;

struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<1>;
  using const_global_cache_tag_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
  using action_list = tmpl::list<>;
};

struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeVolumeData,
                 observers::Actions::ContributeReductionData>;
  using with_these_simple_actions =
      tmpl::list<MockContributeVolumeData, MockContributeReductionData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
  using action_list = tmpl::list<>;
};

struct Metavariables {
  using system = Burgers::System;
  using component_list = tmpl::list<ElementComponent, MockObserverComponent>;
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::AnalyticSolution<Burgers::Solutions::Linear>>;

  struct ObservationType {};
  using element_observation_type = ObservationType;
};

template <typename ObserveEvent>
void test_observe(const std::unique_ptr<ObserveEvent> observe) noexcept {
  Burgers::Solutions::Linear analytic_solution(0.0);
  const ElementComponent::array_index array_index(ElementId<1>(2));
  const std::string element_name =
      get_output(static_cast<ElementId<1>>(array_index));
  const Mesh<1> mesh(5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const double observation_time = 2.0;

  // Fill the variables with some data.  It doesn't matter much what,
  // but integers are nice in that we don't have to worry about
  // roundoff error.
  const tnsr::I<DataVector, 1> coords{{{{4., 3., 2., 1., 0.}}}};
  const Scalar<DataVector> u{{{{1., 2., 3., 4., 5.}}}};
  const Scalar<DataVector> u_analytic =
      tuples::get<Burgers::Tags::U>(analytic_solution.variables(
          coords, observation_time, tmpl::list<Burgers::Tags::U>{}));
  const Scalar<DataVector> u_error{get(u) - get(u_analytic)};

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;

  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<typename MockRuntimeSystem::template MockDistributedObjectsTag<
      ElementComponent>>(dist_objects)
      .emplace(array_index,
               ActionTesting::MockDistributedObject<ElementComponent>{});
  tuples::get<typename MockRuntimeSystem::template MockDistributedObjectsTag<
      MockObserverComponent>>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<MockObserverComponent>{});
  MockRuntimeSystem runner({std::move(analytic_solution)},
                           std::move(dist_objects));

  const auto box =
      db::create<db::AddSimpleTags<Tags::TimeId, Tags::Mesh<1>,
                                   Tags::Coordinates<1, Frame::Inertial>,
                                   Burgers::Tags::U>,
                 db::AddComputeTags<Tags::Time>>(
          TimeId(true, 0, Slab(0., observation_time).end()), mesh, coords, u);

  observe->run(box, runner.cache(), array_index,
               std::add_pointer_t<ElementComponent>{});

  // Process the volume and reduction data
  runner.invoke_queued_simple_action<MockObserverComponent>(0);
  runner.invoke_queued_simple_action<MockObserverComponent>(0);
  CHECK(runner.is_simple_action_queue_empty<MockObserverComponent>(0));

  const auto& volume_results = MockContributeVolumeData::results;
  CHECK(volume_results.observation_id.value() == observation_time);
  CHECK(volume_results.subfile_name == "/element_data");
  CHECK(volume_results.array_component_id ==
        observers::ArrayComponentId(
            std::add_pointer_t<ElementComponent>{},
            Parallel::ArrayIndex<ElementIndex<1>>(array_index)));
  CHECK(volume_results.in_received_tensor_data.size() == 3);
  // gcc 6.4.0 gets confused if we try to capture tensor_data by
  // reference and fails to compile because it wants it to be
  // non-const, so we capture a pointer instead.
  const auto check_component =
      [&element_name, tensor_data = &volume_results.in_received_tensor_data](
          const std::string& component, const DataVector& expected) noexcept {
    CAPTURE(*tensor_data);
    CAPTURE(component);
    const auto it =
        alg::find_if(*tensor_data, [name = element_name + "/" + component](
                                       const TensorComponent& tc) noexcept {
          return tc.name == name;
        });
    CHECK(it != tensor_data->end());
    if (it != tensor_data->end()) {
      CHECK(it->data == expected);
    }
  };
  check_component("InertialCoordinates_x", get<0>(coords));
  check_component("U", get(u));
  check_component("ErrorU", get(u_error));
  CHECK(volume_results.received_extents == mesh.extents());

  const auto& reduction_results = MockContributeReductionData::results;
  CHECK(reduction_results.observation_id.value() == observation_time);
  CHECK(reduction_results.subfile_name == "/element_data");
  CHECK(reduction_results.reduction_names ==
        std::vector<std::string>{"Time", "NumberOfPoints", "ErrorU"});
  CHECK(reduction_results.time == observation_time);
  CHECK(reduction_results.number_of_grid_points ==
        mesh.number_of_grid_points());
  // The rest of the RMS calculation is done later by the writer.
  CHECK(reduction_results.u_error ==
        alg::accumulate(square(get(u_error)), 0.0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Burgers.Observe", "[Unit][Burgers]") {
  test_observe(std::make_unique<Burgers::Events::Observe<>>());

  INFO("create/serialize");
  using EventType = Event<tmpl::list<Burgers::Events::Registrars::Observe>>;
  Parallel::register_derived_classes_with_charm<EventType>();
  const auto factory_event = test_factory_creation<EventType>("  Observe");
  auto serialized_event = serialize_and_deserialize(factory_event);
  test_observe(std::move(serialized_event));
}
