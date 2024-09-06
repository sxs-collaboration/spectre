// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/IteratePunctureField.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/SendToWorldtube.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeElementFacesGridCoordinates.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/IterateAccelerationTerms.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/ReceiveElementData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/UpdateAcceleration.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonChare.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

template <typename Metavariables>
struct MockElementArray {
  using metavariables = Metavariables;
  static constexpr size_t Dim = metavariables::volume_dim;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              db::AddSimpleTags<
                  domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                  domain::Tags::Coordinates<Dim, Frame::Inertial>,
                  Tags::PunctureField<Dim>, gr::Tags::Shift<DataVector, Dim>,
                  gr::Tags::Lapse<DataVector>, ::Tags::TimeStepId,
                  Tags::ParticlePositionVelocity<Dim>, Tags::FaceQuantities,
                  Tags::CurrentIteration>,
              db::AddComputeTags<
                  Tags::FaceCoordinatesCompute<Dim, Frame::Inertial, true>>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::SendToWorldtube, Actions::IteratePunctureField,
                     CurvedScalarWave::Worldtube::Actions::
                         ReceiveWorldtubeData>>>;
};
template <typename Metavariables>
struct MockWorldtubeSingleton {
  using metavariables = Metavariables;
  static constexpr size_t Dim = metavariables::volume_dim;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using variables_tag = ::Tags::Variables<
      tmpl::list<Tags::EvolvedPosition<Dim>, Tags::EvolvedVelocity<Dim>>>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              db::AddSimpleTags<
                  Tags::ElementFacesGridCoordinates<Dim>, ::Tags::TimeStepId,
                  Tags::CurrentIteration, Tags::GeodesicAcceleration<Dim>,
                  CurvedScalarWave::Worldtube::Tags::ParticlePositionVelocity<
                      Dim>,
                  dt_variables_tag>,
              db::AddComputeTags<Tags::BackgroundQuantitiesCompute<Dim>>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::ReceiveElementData,
                     Actions::IterateAccelerationTerms<Metavariables>,
                     ::Actions::MutateApply<UpdateAcceleration>>>>;
  using component_being_mocked = WorldtubeSingleton<Metavariables>;
};

template <size_t Dim>
struct MockMetavariables {
  static constexpr size_t volume_dim = Dim;

  using component_list = tmpl::list<MockWorldtubeSingleton<MockMetavariables>,
                                    MockElementArray<MockMetavariables>>;
  using dg_element_array = MockElementArray<MockMetavariables>;
  using const_global_cache_tags = tmpl::list<
      domain::Tags::Domain<Dim>,
      CurvedScalarWave::Tags::BackgroundSpacetime<gr::Solutions::KerrSchild>,
      Tags::ExcisionSphere<Dim>, Tags::ExpansionOrder, Tags::MaxIterations,
      Tags::Charge, Tags::Mass>;
};

void test_iterations(const size_t max_iterations) {
  CAPTURE(max_iterations);
  static constexpr size_t Dim = 3;
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-10., 10.);
  std::uniform_real_distribution<> pos_dist(2., 10.);
  std::uniform_real_distribution<> vel_dist(-0.2, 0.2);
  using metavars = MockMetavariables<Dim>;
  domain::creators::register_derived_with_charm();
  using element_chare = MockElementArray<metavars>;
  using worldtube_chare = MockWorldtubeSingleton<metavars>;
  const size_t initial_extent = 3;
  const size_t face_size = initial_extent * initial_extent;
  const DataVector used_for_size(face_size);
  const auto quadrature = Spectral::Quadrature::GaussLobatto;
  const double charge = 0.1;
  const double mass = 0.1;
  gr::Solutions::KerrSchild kerr_schild(1., {{0., 0., 0.}}, {{0., 0., 0.}});
  // we create several differently refined shells so a different number of
  // elements sends data
  for (const auto& [expansion_order, initial_refinement, worldtube_radius] :
       cartesian_product(std::array<size_t, 2>{0, 1},
                         std::array<size_t, 2>{0, 1},
                         make_array(0.07, 1., 2.8))) {
    CAPTURE(expansion_order);
    CAPTURE(worldtube_radius);
    CAPTURE(initial_refinement);
    const domain::creators::Sphere shell{worldtube_radius,
                                         3.,
                                         domain::creators::Sphere::Excision{},
                                         initial_refinement,
                                         initial_extent,
                                         true};
    const auto shell_domain = shell.create_domain();
    const auto excision_sphere =
        shell_domain.excision_spheres().at("ExcisionSphere");

    const auto& initial_refinements = shell.initial_refinement_levels();
    const auto& initial_extents = shell.initial_extents();
    tuples::TaggedTuple<
        domain::Tags::Domain<Dim>,
        CurvedScalarWave::Tags::BackgroundSpacetime<gr::Solutions::KerrSchild>,
        Tags::ExcisionSphere<Dim>, Tags::ExpansionOrder, Tags::MaxIterations,
        Tags::Charge, Tags::Mass>
        tuple_of_opts{shell.create_domain(),   kerr_schild,    excision_sphere,
                      expansion_order,         max_iterations, charge,
                      std::make_optional(mass)};
    ActionTesting::MockRuntimeSystem<metavars> runner{std::move(tuple_of_opts)};
    const auto element_ids = initial_element_ids(initial_refinements);
    const auto& blocks = shell_domain.blocks();
    using puncture_field_type =
        Variables<tmpl::list<CurvedScalarWave::Tags::Psi,
                             ::Tags::dt<CurvedScalarWave::Tags::Psi>,
                             ::Tags::deriv<CurvedScalarWave::Tags::Psi,
                                           tmpl::size_t<3>, Frame::Inertial>>>;
    const puncture_field_type puncture_field{face_size, 0.};
    const Time dummy_time{{1., 2.}, {1, 2}};
    const TimeStepId dummy_time_step_id{true, 123, dummy_time};

    const auto particle_position =
        make_with_random_values<tnsr::I<double, Dim>>(make_not_null(&generator),
                                                      pos_dist, 1);
    const auto particle_velocity =
        make_with_random_values<tnsr::I<double, Dim>>(make_not_null(&generator),
                                                      vel_dist, 1);
    const std::array<tnsr::I<double, Dim>, 2> particle_pos_vel{
        particle_position, particle_velocity};
    std::vector<ElementId<Dim>> abutting_element_ids{};
    std::vector<ElementId<Dim>> non_abutting_element_ids{};

    for (const auto& element_id : element_ids) {
      const auto& my_block = blocks.at(element_id.block_id());
      auto element = domain::Initialization::create_initial_element(
          element_id, my_block, initial_refinements);
      auto mesh = domain::Initialization::create_initial_mesh(
          initial_extents, element_id, quadrature);
      const ElementMap element_map(element_id,
                                   my_block.stationary_map().get_clone());
      const auto logical_coords = logical_coordinates(mesh);
      const auto inertial_coords = element_map(logical_coords);
      auto lapse_and_shift = kerr_schild.variables(
          inertial_coords, 0.,
          tmpl::list<gr::Tags::Lapse<DataVector>,
                     gr::Tags::Shift<DataVector, Dim, Frame::Inertial>>{});
      const bool is_abutting =
          excision_sphere.abutting_direction(element_id).has_value();
      using face_quantities_type =
          Variables<tmpl::list<CurvedScalarWave::Tags::Psi,
                               ::Tags::dt<CurvedScalarWave::Tags::Psi>,
                               gr::surfaces::Tags::AreaElement<DataVector>>>;
      std::optional<face_quantities_type> optional_face_quantities =
          is_abutting
              ? std::make_optional<face_quantities_type>(
                    make_with_random_values<face_quantities_type>(
                        make_not_null(&generator), dist, DataVector(face_size)))
              : std::nullopt;
      std::optional<puncture_field_type> optional_puncture_field =
          is_abutting ? std::make_optional<puncture_field_type>(puncture_field)
                      : std::nullopt;
      ActionTesting::emplace_array_component_and_initialize<element_chare>(
          &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
          element_id,
          {std::move(element), std::move(mesh), inertial_coords,
           std::move(optional_puncture_field),
           std::move(get<gr::Tags::Shift<DataVector, Dim, Frame::Inertial>>(
               lapse_and_shift)),
           std::move(get<gr::Tags::Lapse<DataVector>>(lapse_and_shift)),
           dummy_time_step_id, particle_pos_vel, optional_face_quantities,
           static_cast<size_t>(0)});
      if (is_abutting) {
        abutting_element_ids.push_back(element_id);

      } else {
        non_abutting_element_ids.push_back(element_id);
      }
    }

    std::unordered_map<ElementId<Dim>, tnsr::I<DataVector, Dim, Frame::Grid>>
        element_faces_grid_coords{};
    Initialization::InitializeElementFacesGridCoordinates<Dim>::apply(
        make_not_null(&element_faces_grid_coords), initial_extents,
        initial_refinements, quadrature, shell_domain, excision_sphere);
    // we set the geodesic acceleration to zero, so the particle acceleration is
    // just given by the self force
    const tnsr::I<double, Dim> geodesic_acc(0.);
    ActionTesting::emplace_singleton_component_and_initialize<worldtube_chare>(
        &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
        {element_faces_grid_coords, dummy_time_step_id, static_cast<size_t>(0),
         geodesic_acc, particle_pos_vel,
         MockWorldtubeSingleton<
             MockMetavariables<Dim>>::dt_variables_tag::type{}});
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

    // check that the non-abutting element_ids can just keep iterating
    for (const auto& element_id : non_abutting_element_ids) {
      CHECK(ActionTesting::get_next_action_index<element_chare>(
                runner, element_id) == 0);
      // SendToWorldtube
      CHECK(ActionTesting::next_action_if_ready<element_chare>(
          make_not_null(&runner), element_id));
      CHECK(ActionTesting::get_next_action_index<element_chare>(
                runner, element_id) == 1);
      // IteratePunctureField
      CHECK(ActionTesting::next_action_if_ready<element_chare>(
          make_not_null(&runner), element_id));
      CHECK(ActionTesting::get_next_action_index<element_chare>(
                runner, element_id) == 2);
      // ReceiveWorldtubeData
      CHECK(ActionTesting::next_action_if_ready<element_chare>(
          make_not_null(&runner), element_id));
      CHECK(ActionTesting::get_next_action_index<element_chare>(
                runner, element_id) == 0);
    }

    // ReceiveElementData should not be ready yet as the worldtube has not
    // received any data
    CHECK(not ActionTesting::next_action_if_ready<worldtube_chare>(
        make_not_null(&runner), 0));

    for (const auto& element_id : abutting_element_ids) {
      const auto& element_iteration =
          ActionTesting::get_databox_tag<element_chare, Tags::CurrentIteration>(
              runner, element_id);
      CHECK(element_iteration == 0);
      // SendToWorldtube called on all elements
      ActionTesting::next_action<element_chare>(make_not_null(&runner),
                                                element_id);
      // expecting data from the worldtube now
      CHECK(not ActionTesting::next_action_if_ready<element_chare>(
          make_not_null(&runner), element_id));
    }

    for (size_t current_iteration = 1; current_iteration + 1 <= max_iterations;
         ++current_iteration) {
      CAPTURE(current_iteration);
      using inbox_tag = Tags::SphericalHarmonicsInbox<Dim>;
      const auto& worldtube_inbox =
          ActionTesting::get_inbox_tag<worldtube_chare, inbox_tag>(runner, 0);
      CHECK(worldtube_inbox.count(dummy_time_step_id));
      auto time_step_data = worldtube_inbox.at(dummy_time_step_id);
      // these are all the element ids of elements abutting the worldtube, we
      // check that these are the ones that were sent.
      for (const auto& [element_id, _] : element_faces_grid_coords) {
        CHECK(time_step_data.count(element_id));
        time_step_data.erase(element_id);
      }
      // Check that have received only data from elements abutting the worldtube
      CHECK(time_step_data.empty());
      // ReceiveElementData
      CHECK(ActionTesting::next_action_if_ready<worldtube_chare>(
          make_not_null(&runner), 0));
      const auto& singleton_iteration =
          ActionTesting::get_databox_tag<worldtube_chare,
                                         Tags::CurrentIteration>(runner, 0);
      CHECK(singleton_iteration == current_iteration);
      CHECK(worldtube_inbox.empty());
      // IterateAccelerationTerms
      CHECK(ActionTesting::next_action_if_ready<worldtube_chare>(
          make_not_null(&runner), 0));
      // expecting data from the elements now which is not sent yet
      CHECK(not ActionTesting::next_action_if_ready<worldtube_chare>(
          make_not_null(&runner), 0));

      const auto& dt_psi_monopole = ActionTesting::get_databox_tag<
          worldtube_chare, Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>,
                                                0, Dim, Frame::Inertial>>(
          runner, 0);
      const auto& psi_dipole = ActionTesting::get_databox_tag<
          worldtube_chare,
          Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Inertial>>(
          runner, 0);
      const auto& background =
          ActionTesting::get_databox_tag<worldtube_chare,
                                         Tags::BackgroundQuantities<Dim>>(
              runner, 0);

      const auto self_force_acc = self_force_acceleration(
          dt_psi_monopole, psi_dipole, particle_velocity, charge, mass,
          get<gr::Tags::InverseSpacetimeMetric<double, Dim>>(background),
          get<Tags::TimeDilationFactor>(background));
      for (const auto& element_id : abutting_element_ids) {
        const auto& self_force_inbox =
            ActionTesting::get_inbox_tag<element_chare,
                                         Tags::SelfForceInbox<Dim>>(runner,
                                                                    element_id);
        CHECK(self_force_inbox.count(dummy_time_step_id));
        for (size_t i = 0; i < Dim; ++i) {
          CHECK(get(self_force_inbox.at(dummy_time_step_id))[i] ==
                self_force_acc.get(i));
        }
        const std::string inbox_output =
            Tags::SelfForceInbox<Dim>::output_inbox(self_force_inbox, 2);
        const std::string expected_inbox_output =
            MakeString{} << std::scientific << std::setprecision(16)
                         << "  SelfForceInbox:\n"
                         << "   Time: " << dummy_time_step_id << "\n";
        CHECK(inbox_output == expected_inbox_output);
        // IteratePunctureField
        CHECK(ActionTesting::next_action_if_ready<element_chare>(
            make_not_null(&runner), element_id));
        CHECK(self_force_inbox.empty());
        const auto& element_iteration =
            ActionTesting::get_databox_tag<element_chare,
                                           Tags::CurrentIteration>(runner,
                                                                   element_id);
        CHECK(element_iteration == current_iteration);
        if (current_iteration > 0) {
          CHECK(
              ActionTesting::get_databox_tag<element_chare,
                                             Tags::IteratedPunctureField<Dim>>(
                  runner, element_id)
                  .has_value());
        }
        // SendToWorldtube
        CHECK(ActionTesting::next_action_if_ready<element_chare>(
            make_not_null(&runner), element_id));
        CHECK(not ActionTesting::next_action_if_ready<element_chare>(
            make_not_null(&runner), element_id));
      }
    }
    CHECK(ActionTesting::get_next_action_index<worldtube_chare>(runner, 0) ==
          0);
    // ReceiveElementData
    CHECK(ActionTesting::next_action_if_ready<worldtube_chare>(
        make_not_null(&runner), 0));
    // UpdateAcceleration should be queued now
    CHECK(ActionTesting::get_next_action_index<worldtube_chare>(runner, 0) ==
          2);
    // iterations should have reset for singleton
    const auto& singleton_iteration =
        ActionTesting::get_databox_tag<worldtube_chare, Tags::CurrentIteration>(
            runner, 0);
    CHECK(singleton_iteration == 0);
    for (const auto& element_id : abutting_element_ids) {
      // Should be at ReceiveWorldtubeData now
      CHECK(ActionTesting::get_next_action_index<element_chare>(
                runner, element_id) == 2);
      // iterations should have reset for elements
      const auto& element_iteration =
          ActionTesting::get_databox_tag<element_chare, Tags::CurrentIteration>(
              runner, element_id);
      CHECK(element_iteration == 0);
    }
  }
}

SPECTRE_TEST_CASE("Unit.CurvedScalarWave.Worldtube.Iterations", "[Unit]") {
  test_iterations(0);
  test_iterations(1);
  test_iterations(2);
  test_iterations(5);
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
