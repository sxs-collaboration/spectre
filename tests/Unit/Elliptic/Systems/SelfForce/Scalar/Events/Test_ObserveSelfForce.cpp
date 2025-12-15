// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
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
#include "Helpers/IO/Observers/MockWriteReductionDataRow.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<2>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct Metavariables {
  using component_list =
      tmpl::list<ElementComponent<Metavariables>,
                 TestHelpers::observers::MockObserverWriter<Metavariables>>;
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
  const ElementId<2> element_id{0, {{{0, 0}, {0, 0}}}};

  using mock_observer_writer =
      TestHelpers::observers::MockObserverWriter<Metavariables>;
  using element_component = ElementComponent<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_nodegroup_component_and_initialize<
      mock_observer_writer>(make_not_null(&runner), {});
  ActionTesting::emplace_component<element_component>(&runner, element_id);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Just testing that the event runs and writes the expected data. Actual
  // numbers are tested in the scalar self-force input-file test.
  const Scalar<ComplexDataVector> field{mesh.number_of_grid_points(), 0.};
  const ElementMap<2, Frame::Inertial> element_map{
      element_id, domain.blocks()[element_id.block_id()]};
  const auto inv_jacobian = element_map.inv_jacobian(logical_coordinates(mesh));
  auto box = db::create<tmpl::list<
      elliptic::Tags::Background<elliptic::analytic_data::Background>,
      domain::Tags::Domain<2>, domain::Tags::Mesh<2>,
      domain::Tags::InverseJacobian<2, Frame::ElementLogical, Frame::Inertial>,
      ScalarSelfForce::Tags::MMode>>(
      std::unique_ptr<elliptic::analytic_data::Background>(
          std::make_unique<ScalarSelfForce::AnalyticData::CircularOrbit>(
              circular_orbit)),
      domain_creator.create_domain(), mesh, inv_jacobian, field);
  auto obs_box =
      make_observation_box<db::AddComputeTags<>>(make_not_null(&box));

  const ScalarSelfForce::Events::ObserveSelfForce event{};
  event.run(make_not_null(&obs_box),
            ActionTesting::cache<element_component>(runner, element_id),
            element_id, std::add_pointer_t<element_component>{},
            {"Iteration", 1.0});

  ActionTesting::invoke_queued_threaded_action<mock_observer_writer>(
      make_not_null(&runner), 0);
  const auto& mock_h5_file = ActionTesting::get_databox_tag<
      mock_observer_writer, TestHelpers::observers::MockReductionFileTag>(
      runner, 0);
  const auto& dat_file = mock_h5_file.get_dat("/SelfForce");
  const auto& data = dat_file.get_data();
  REQUIRE(data.rows() == 1);
  REQUIRE(data.columns() == 6);
  CHECK(data(0, 0) == 1.0);                           // IterationId
  CHECK(data(0, 1) == mesh.number_of_grid_points());  // NumberOfGridPoints
  CHECK(data(0, 2) == 0.0);                           // Re(SelfForce_r)
  CHECK(data(0, 3) == 0.0);                           // Im(SelfForce_r)
  CHECK(data(0, 4) == 0.0);                           // Re(SelfForce_theta)
  CHECK(data(0, 5) == 0.0);                           // Im(SelfForce_theta)
}
