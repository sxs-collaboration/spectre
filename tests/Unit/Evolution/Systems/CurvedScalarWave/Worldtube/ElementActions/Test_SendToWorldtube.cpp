// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <unordered_map>

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
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/SendToWorldtube.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeElementFacesGridCoordinates.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/ReceiveElementData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonChare.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
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
                  domain::Tags::Coordinates<Dim, Frame::Grid>,
                  Tags::PunctureField<Dim>, gr::Tags::Shift<DataVector, Dim>,
                  gr::Tags::Lapse<DataVector>,
                  domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                Frame::Grid>,
                  typename CurvedScalarWave::System<Dim>::variables_tag,
                  domain::Tags::MeshVelocity<Dim>, ::Tags::TimeStepId>,
              db::AddComputeTags<
                  Tags::FaceCoordinatesCompute<Dim, Frame::Grid, true>>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::SendToWorldtube>>>;
};
template <typename Metavariables>
struct MockWorldtubeSingleton {
  using metavariables = Metavariables;
  static constexpr size_t Dim = metavariables::volume_dim;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              db::AddSimpleTags<Tags::ElementFacesGridCoordinates<Dim>,
                                ::Tags::TimeStepId>,
              db::AddComputeTags<>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::ReceiveElementData>>>;
  using component_being_mocked = WorldtubeSingleton<Metavariables>;
};

template <size_t Dim>
struct MockMetavariables {
  static constexpr size_t volume_dim = Dim;

  using component_list = tmpl::list<MockWorldtubeSingleton<MockMetavariables>,
                                    MockElementArray<MockMetavariables>>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<Dim>, Tags::ExcisionSphere<Dim>,
                 Tags::ExpansionOrder>;
};

SPECTRE_TEST_CASE("Unit.CurvedScalarWave.Worldtube.SendToWorldtube", "[Unit]") {
  static constexpr size_t Dim = 3;
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-10., 10.);
  using metavars = MockMetavariables<Dim>;
  domain::creators::register_derived_with_charm();
  using element_chare = MockElementArray<metavars>;
  using worldtube_chare = MockWorldtubeSingleton<metavars>;
  const size_t initial_extent = 10;
  const size_t face_size = initial_extent * initial_extent;
  const auto quadrature = Spectral::Quadrature::GaussLobatto;
  // we create several differently refined shells so a different number of
  // elements sends data
  for (const auto& [expansion_order, initial_refinement, worldtube_radius] :
       cartesian_product(std::array<size_t, 3>{0, 1, 2},
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
    tuples::TaggedTuple<domain::Tags::Domain<Dim>, Tags::ExcisionSphere<Dim>,
                        Tags::ExpansionOrder>
        tuple_of_opts{shell.create_domain(), excision_sphere, expansion_order};
    ActionTesting::MockRuntimeSystem<metavars> runner{std::move(tuple_of_opts)};
    const auto element_ids = initial_element_ids(initial_refinements);
    const auto& blocks = shell_domain.blocks();

    using puncture_field_type =
        Variables<tmpl::list<CurvedScalarWave::Tags::Psi,
                             ::Tags::dt<CurvedScalarWave::Tags::Psi>,
                             ::Tags::deriv<CurvedScalarWave::Tags::Psi,
                                           tmpl::size_t<3>, Frame::Inertial>>>;

    // The puncture field will get subtracted from the DG field. Here, we set
    // the puncture field to 0, so psi and dt_psi are integrated directly
    // and we can check the analytical result.
    const puncture_field_type puncture_field{face_size, 0.};
    const double psi_coefs_0 = dist(generator);
    const double pi_coefs_0 = dist(generator);
    const auto psi_coefs_1 =
        make_with_random_values<tnsr::i<double, Dim, Frame::Grid>>(
            make_not_null(&generator), dist, 0.);
    const auto pi_coefs_1 =
        make_with_random_values<tnsr::i<double, Dim, Frame::Grid>>(
            make_not_null(&generator), dist, 0.);
    auto psi_coefs_2 =
        make_with_random_values<tnsr::ii<double, Dim, Frame::Grid>>(
            make_not_null(&generator), dist, 0.);
    auto pi_coefs_2 =
        make_with_random_values<tnsr::ii<double, Dim, Frame::Grid>>(
            make_not_null(&generator), dist, 0.);
    double psi_coefs_2_trace = 0.;
    double pi_coefs_2_trace = 0.;

    for (size_t i = 0; i < Dim; ++i) {
      psi_coefs_2_trace += psi_coefs_2.get(i, i);
      pi_coefs_2_trace += pi_coefs_2.get(i, i);
    }
    const Time dummy_time{{1., 2.}, {1, 2}};
    const TimeStepId dummy_time_step_id{true, 123, dummy_time};

    for (const auto& element_id : element_ids) {
      const auto& my_block = blocks.at(element_id.block_id());
      auto element = domain::Initialization::create_initial_element(
          element_id, my_block, initial_refinements);
      auto mesh = domain::Initialization::create_initial_mesh(
          initial_extents, element_id, quadrature);
      const ElementMap element_map(
          element_id, my_block.stationary_map().get_to_grid_frame());
      const auto logical_coords = logical_coordinates(mesh);
      const auto grid_coords = element_map(logical_coords);
      auto grid_inv_jacobian = element_map.inv_jacobian(logical_coords);
      const size_t grid_size = mesh.number_of_grid_points();
      // we set lapse and shift to Minkowski so dt Psi = - Pi, and the value we
      // pass in for Pi will get integrated directly
      Scalar<DataVector> lapse(grid_size, 1.);
      tnsr::I<DataVector, Dim, Frame::Inertial> shift(grid_size, 0.);
      typename CurvedScalarWave::System<Dim>::variables_tag::type evolved_vars(
          grid_size, 0.);
      const bool is_abutting =
          excision_sphere.abutting_direction(element_id).has_value();
      get(get<CurvedScalarWave::Tags::Psi>(evolved_vars)) = psi_coefs_0;
      get(get<CurvedScalarWave::Tags::Pi>(evolved_vars)) = pi_coefs_0;
      if (expansion_order > 0) {
        for (size_t i = 0; i < Dim; ++i) {
          get(get<CurvedScalarWave::Tags::Psi>(evolved_vars)) +=
              psi_coefs_1.get(i) * grid_coords.get(i);
          get(get<CurvedScalarWave::Tags::Pi>(evolved_vars)) +=
              pi_coefs_1.get(i) * grid_coords.get(i);
        }
      }
      if (expansion_order > 1) {
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t j = 0; j < Dim; ++j) {
            get(get<CurvedScalarWave::Tags::Psi>(evolved_vars)) +=
                psi_coefs_2.get(i, j) * grid_coords.get(i) * grid_coords.get(j);
            get(get<CurvedScalarWave::Tags::Pi>(evolved_vars)) +=
                pi_coefs_2.get(i, j) * grid_coords.get(i) * grid_coords.get(j);
          }
        }
      }
      std::optional<puncture_field_type> optional_puncture_field =
          is_abutting ? std::make_optional<puncture_field_type>(puncture_field)
                      : std::nullopt;
      // we set the mesh_velocity to zero so we can recover the exact dt value
      std::optional<tnsr::I<DataVector, Dim>> mesh_velocity =
          is_abutting
              ? std::make_optional<tnsr::I<DataVector, Dim>>(face_size, 0.)
              : std::nullopt;

      ActionTesting::emplace_array_component_and_initialize<element_chare>(
          &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
          element_id,
          {std::move(element), std::move(mesh), grid_coords,
           std::move(optional_puncture_field), std::move(shift),
           std::move(lapse), std::move(grid_inv_jacobian), evolved_vars,
           std::move(mesh_velocity), dummy_time_step_id});
    }

    std::unordered_map<ElementId<Dim>, tnsr::I<DataVector, Dim, Frame::Grid>>
        element_faces_grid_coords{};
    Initialization::InitializeElementFacesGridCoordinates<Dim>::apply(
        make_not_null(&element_faces_grid_coords), initial_extents,
        initial_refinements, quadrature, shell_domain, excision_sphere);

    ActionTesting::emplace_singleton_component_and_initialize<worldtube_chare>(
        &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
        {element_faces_grid_coords, dummy_time_step_id});

    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

    // ReceiveElementData should not be ready yet as the worldtube has not
    // received any data
    CHECK(not ActionTesting::next_action_if_ready<worldtube_chare>(
        make_not_null(&runner), 0));
    // SendToWorldtube called on all elements
    for (const auto& element_id : element_ids) {
      ActionTesting::next_action<element_chare>(make_not_null(&runner),
                                                element_id);
      if (excision_sphere.abutting_direction(element_id).has_value()) {
        const auto& regular_field_advective_term =
            ActionTesting::get_databox_tag<
                element_chare, Tags::RegularFieldAdvectiveTerm<Dim>>(
                runner, element_id);
        CHECK_ITERABLE_APPROX(regular_field_advective_term,
                              Scalar<DataVector>(face_size, 0.));
      }
    }

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
    // ReceiveElementData called
    CHECK(ActionTesting::next_action_if_ready<worldtube_chare>(
        make_not_null(&runner), 0));
    CHECK(worldtube_inbox.empty());
    const auto& psi_monopole_worldtube = ActionTesting::get_databox_tag<
        worldtube_chare,
        Stf::Tags::StfTensor<Tags::PsiWorldtube, 0, Dim, Frame::Grid>>(runner,
                                                                       0);
    const auto& dt_psi_monopole_worldtube = ActionTesting::get_databox_tag<
        worldtube_chare, Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0,
                                              Dim, Frame::Grid>>(runner, 0);
    // the integral is over a low resolution DG grid which introduces a large
    // error
    Approx apprx = Approx::custom().epsilon(1e-8).scale(1.0);
    if (expansion_order < 2) {
      CHECK(get(psi_monopole_worldtube) == apprx(psi_coefs_0));
      CHECK(get(dt_psi_monopole_worldtube) == -apprx(pi_coefs_0));
    }
    if (expansion_order > 0) {
      const auto& psi_dipole_worldtube = ActionTesting::get_databox_tag<
          worldtube_chare,
          Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Grid>>(runner,
                                                                         0);
      const auto& dt_psi_dipole_worldtube = ActionTesting::get_databox_tag<
          worldtube_chare, Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>,
                                                1, Dim, Frame::Grid>>(runner,
                                                                      0);
      CHECK_ITERABLE_CUSTOM_APPROX(psi_dipole_worldtube, psi_coefs_1, apprx);
      for (size_t i = 0; i < Dim; ++i) {
        CHECK(dt_psi_dipole_worldtube.get(i) == apprx(-pi_coefs_1.get(i)));
      }
      if (expansion_order > 1) {
        // the trace of the second order coefficients gets absorbed by the
        // monopole
        const double expected_psi_monopole =
            psi_coefs_0 + square(worldtube_radius) * psi_coefs_2_trace / 3.;
        const double expected_pi_monopole =
            pi_coefs_0 + square(worldtube_radius) * pi_coefs_2_trace / 3.;
        CHECK(get(psi_monopole_worldtube) == apprx(expected_psi_monopole));
        CHECK(get(dt_psi_monopole_worldtube) == -apprx(expected_pi_monopole));
        const auto& psi_quadrupole_worldtube = ActionTesting::get_databox_tag<
            worldtube_chare,
            Stf::Tags::StfTensor<Tags::PsiWorldtube, 2, Dim, Frame::Grid>>(
            runner, 0);
        const auto& dt_psi_quadrupole_worldtube =
            ActionTesting::get_databox_tag<
                worldtube_chare,
                Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 2, Dim,
                                     Frame::Grid>>(runner, 0);
        tnsr::ii<double, Dim, Frame::Grid> expected_psi_quadrupole{};
        tnsr::ii<double, Dim, Frame::Grid> expected_dt_psi_quadrupole{};
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t j = 0; j < Dim; ++j) {
            expected_psi_quadrupole.get(i, j) = psi_coefs_2.get(i, j);
            expected_dt_psi_quadrupole.get(i, j) = -pi_coefs_2.get(i, j);
          }
          expected_psi_quadrupole.get(i, i) -= psi_coefs_2_trace / 3.;
          expected_dt_psi_quadrupole.get(i, i) += pi_coefs_2_trace / 3.;
        }
        CHECK_ITERABLE_CUSTOM_APPROX(psi_quadrupole_worldtube,
                                     expected_psi_quadrupole, apprx);
        CHECK_ITERABLE_CUSTOM_APPROX(dt_psi_quadrupole_worldtube,
                                     expected_dt_psi_quadrupole, apprx);
      }
    }
  }
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
