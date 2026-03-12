// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Events/ObserveSelfForce.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct MockContributeReductionData {
  using ReductionData =
      ScalarSelfForce::Events::ObserveSelfForce<>::ReductionData;

  struct Results {
    observers::ObservationId observation_id{};
    std::string subfile_name{};
    std::vector<std::string> legend{};
    double iteration_id{};
    size_t num_grid_points{};
    size_t num_contributing_elements{};
    std::vector<double> self_force{};
  };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static Results results;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::optional<ReductionData> combined_reduction_data;

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex, typename... Ts>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    Parallel::ArrayComponentId /*sender_array_id*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& legend,
                    Parallel::ReductionData<Ts...>&& local_reduction_data) {
    if (not MockContributeReductionData::combined_reduction_data.has_value()) {
      MockContributeReductionData::combined_reduction_data.emplace(
          std::move(local_reduction_data));
    } else {
      MockContributeReductionData::combined_reduction_data->combine(
          std::move(local_reduction_data));
    }
    auto reduction_data = *MockContributeReductionData::combined_reduction_data;
    reduction_data.finalize();
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.legend = legend;
    results.iteration_id = std::get<0>(reduction_data.data());
    results.num_grid_points = std::get<1>(reduction_data.data());
    results.num_contributing_elements = std::get<2>(reduction_data.data());
    results.self_force = {
        std::get<3>(reduction_data.data()), std::get<4>(reduction_data.data()),
        std::get<5>(reduction_data.data()), std::get<6>(reduction_data.data())};
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
MockContributeReductionData::Results MockContributeReductionData::results{};
std::optional<MockContributeReductionData::ReductionData>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    MockContributeReductionData::combined_reduction_data{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<2>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeReductionData>;
  using with_these_simple_actions = tmpl::list<MockContributeReductionData>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        Event, tmpl::list<ScalarSelfForce::Events::ObserveSelfForce<>>>>;
  };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.ScalarSelfForce.Events.ObserveSelfForce",
                  "[Unit][Elliptic]") {
  const ScalarSelfForce::AnalyticData::CircularOrbit circular_orbit{
      1.0, 0.0, 10.0, 2, std::nullopt};

  const double puncture_r_star = get<0>(circular_orbit.puncture_position());
  const domain::creators::Rectilinear<2> domain_creator{
      {{puncture_r_star - 1.0, -0.5}},
      {{puncture_r_star + 1.0, 0.5}},
      {{0, 0}},
      {{4, 4}}};
  const auto domain = domain_creator.create_domain();
  const Mesh<2> mesh{4_st, Spectral::Basis::Legendre,
                     Spectral::Quadrature::Gauss};
  const ElementId<2> element_id_a{0, {{{1, 0}, {0, 0}}}};
  const ElementId<2> element_id_b{0, {{{1, 1}, {0, 0}}}};

  using mock_observer_writer = MockObserverComponent<Metavariables>;
  using element_component = ElementComponent<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_group_component<mock_observer_writer>(&runner);
  ActionTesting::emplace_component<element_component>(&runner, element_id_a);
  ActionTesting::emplace_component<element_component>(&runner, element_id_b);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Just testing that the event runs and writes the expected data. Actual
  // numbers are tested in the scalar self-force input-file test.
  const Scalar<ComplexDataVector> field{mesh.number_of_grid_points(), 0.};
  const ScalarSelfForce::Events::ObserveSelfForce event{};
  for (const auto& element_id : std::array{element_id_a, element_id_b}) {
    const ElementMap<2, Frame::Inertial> element_map{
        element_id, domain.blocks()[element_id.block_id()]};
    const auto inv_jacobian =
        element_map.inv_jacobian(logical_coordinates(mesh));
    auto box = db::create<tmpl::list<
        elliptic::Tags::Background<elliptic::analytic_data::Background>,
        domain::Tags::Domain<2>, domain::Tags::Mesh<2>,
        domain::Tags::InverseJacobian<2, Frame::ElementLogical,
                                      Frame::Inertial>,
        ScalarSelfForce::Tags::MMode>>(
        std::unique_ptr<elliptic::analytic_data::Background>(
            std::make_unique<ScalarSelfForce::AnalyticData::CircularOrbit>(
                circular_orbit)),
        domain_creator.create_domain(), mesh, inv_jacobian, field);
    auto obs_box =
        make_observation_box<db::AddComputeTags<>>(make_not_null(&box));
    event.run(make_not_null(&obs_box),
              ActionTesting::cache<element_component>(runner, element_id),
              element_id, std::add_pointer_t<element_component>{},
              {"Iteration", 1.0});
  }

  runner.template invoke_queued_simple_action<mock_observer_writer>(0);
  runner.template invoke_queued_simple_action<mock_observer_writer>(0);
  CHECK(runner.template is_simple_action_queue_empty<mock_observer_writer>(0));

  const auto& results = MockContributeReductionData::results;
  CHECK(results.observation_id.value() == 1.0);
  CHECK(results.observation_id.observation_key() ==
        observers::ObservationKey("SelfForce.dat"));
  CHECK(results.subfile_name == "SelfForce.dat");
  CHECK(results.legend ==
        std::vector<std::string>{"IterationId", "NumGridPoints",
                                 "NumContributingElements", "Re(SelfForce_r)",
                                 "Im(SelfForce_r)", "Re(SelfForce_theta)",
                                 "Im(SelfForce_theta)"});
  CHECK(results.iteration_id == 1.0);
  CHECK(results.num_grid_points == 2 * mesh.number_of_grid_points());
  CHECK(results.num_contributing_elements == 2_st);
  CHECK(results.self_force == std::vector<double>{0.0, 0.0, 0.0, 0.0});
}
