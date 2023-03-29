// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
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
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeElementFacesGridCoordinates.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/SendToElements.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonChare.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
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
                  Tags::PunctureField<Dim>,
                  gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
                  gr::Tags::Lapse<DataVector>,
                  domain::Tags::InverseJacobian<Dim, Frame::Grid,
                                                Frame::Inertial>,
                  ::Tags::TimeStepId, Tags::RegularFieldAdvectiveTerm<Dim>,
                  Tags::WorldtubeSolution<Dim>>,
              db::AddComputeTags<>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::ReceiveWorldtubeData>>>;
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
              db::AddSimpleTags<
                  Tags::ElementFacesGridCoordinates<Dim>, ::Tags::TimeStepId,
                  Stf::Tags::StfTensor<Tags::PsiWorldtube, 0, Dim, Frame::Grid>,
                  Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                                       Frame::Grid>>,
              db::AddComputeTags<>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::SendToElements<Metavariables>>>>;
  using component_being_mocked = WorldtubeSingleton<Metavariables>;
};

template <size_t Dim>
struct MockMetavariables {
  static constexpr size_t volume_dim = Dim;
  using dg_element_array = MockElementArray<MockMetavariables>;
  using component_list = tmpl::list<MockWorldtubeSingleton<MockMetavariables>,
                                    MockElementArray<MockMetavariables>>;
  using const_global_cache_tags =
      tmpl::list<Tags::ExcisionSphere<Dim>, Tags::ExpansionOrder>;
};

SPECTRE_TEST_CASE("Unit.CurvedScalarWave.Worldtube.ReceiveWorldtubeData",
                  "[Unit]") {
  static constexpr size_t Dim = 3;
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-10., 10.);
  using metavars = MockMetavariables<Dim>;
  domain::creators::register_derived_with_charm();
  using element_chare = MockElementArray<metavars>;
  using worldtube_chare = MockWorldtubeSingleton<metavars>;
  const size_t initial_extent = 8;
  const size_t face_size = initial_extent * initial_extent;
  const auto quadrature = Spectral::Quadrature::GaussLobatto;
  const size_t expansion_order = 0;
  // we create several differently refined shells so a different number of
  // elements sends data
  for (const auto& [initial_refinement, worldtube_radius] : cartesian_product(
           std::array<size_t, 3>{0, 1, 2}, make_array(0.07, 1., 2.8))) {
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
    ActionTesting::MockRuntimeSystem<metavars> runner{
        {excision_sphere, expansion_order}};
    const auto element_ids = initial_element_ids(initial_refinements);
    const auto& blocks = shell_domain.blocks();

    using puncture_field_type =
        Variables<tmpl::list<CurvedScalarWave::Tags::Psi,
                             ::Tags::dt<CurvedScalarWave::Tags::Psi>,
                             ::Tags::deriv<CurvedScalarWave::Tags::Psi,
                                           tmpl::size_t<3>, Frame::Inertial>>>;

    // The puncture field will get subtracted from the DG field. Here, we set
    // the puncture field to 0, so psi and dt_psi are passed on directly
    // and we can check the analytical result.
    const puncture_field_type puncture_field{face_size, 0.};
    const double psi_value = dist(generator);
    const double dt_psi_value = dist(generator);
    const Time dummy_time{{1., 2.}, {1, 2}};
    const TimeStepId dummy_time_step_id{true, 123, dummy_time};

    for (const auto& element_id : element_ids) {
      auto mesh = domain::Initialization::create_initial_mesh(
          initial_extents, element_id, quadrature);
      const size_t grid_size = mesh.number_of_grid_points();
      const auto& my_block = blocks.at(element_id.block_id());
      auto element = domain::Initialization::create_initial_element(
          element_id, my_block, initial_refinements);
      // since the spatial derivative is zero at zeroth order this will not be
      // used
      auto grid_inv_jacobian =
          InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>(
              grid_size, 1.);
      // we set lapse and shift to Minkowski so dt Psi = - Pi
      Scalar<DataVector> lapse(grid_size, 1.);
      tnsr::I<DataVector, Dim, Frame::Inertial> shift(grid_size, 0.);
      // not initialized, the action does that for us
      typename CurvedScalarWave::System<Dim>::variables_tag::type
          worldtube_solution{};
      std::optional<puncture_field_type> optional_puncture_field =
          excision_sphere.abutting_direction(element_id).has_value()
              ? std::make_optional<puncture_field_type>(puncture_field)
              : std::nullopt;

      Scalar<DataVector> advective_term(face_size, 0.);
      ActionTesting::emplace_array_component_and_initialize<element_chare>(
          &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
          element_id,
          {std::move(element), std::move(mesh),
           std::move(optional_puncture_field), std::move(shift),
           std::move(lapse), std::move(grid_inv_jacobian), dummy_time_step_id,
           std::move(advective_term), worldtube_solution});
    }

    std::unordered_map<ElementId<Dim>, tnsr::I<DataVector, Dim, Frame::Grid>>
        element_faces_grid_coords{};
    Initialization::InitializeElementFacesGridCoordinates<Dim>::apply(
        make_not_null(&element_faces_grid_coords), initial_extents,
        initial_refinements, quadrature, shell_domain, excision_sphere);

    ActionTesting::emplace_singleton_component_and_initialize<worldtube_chare>(
        &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
        {element_faces_grid_coords, dummy_time_step_id,
         Scalar<double>(psi_value), Scalar<double>(dt_psi_value)});

    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

    // If it is an abutting element, we don't have data yet and should not be
    // ready. Else, there is nothing to do so it's ready.
    for (const auto& element_id : element_ids) {
      if (element_faces_grid_coords.count(element_id)) {
        CHECK(not ActionTesting::next_action_if_ready<element_chare>(
            make_not_null(&runner), element_id));
      } else {
        CHECK(ActionTesting::next_action_if_ready<element_chare>(
            make_not_null(&runner), element_id));
      }
    }
    // SendToElements
    ActionTesting::next_action<worldtube_chare>(make_not_null(&runner), 0);

    for (const auto& element_id : element_ids) {
      using inbox_tag = Tags::RegularFieldInbox<Dim>;
      const auto& element_inbox =
          ActionTesting::get_inbox_tag<element_chare, inbox_tag>(runner,
                                                                 element_id);
      if (element_faces_grid_coords.count(element_id)) {
        CHECK(element_inbox.count(dummy_time_step_id));
        const auto& inbox_data = element_inbox.at(dummy_time_step_id);
        CHECK_ITERABLE_APPROX(get(get<CurvedScalarWave::Tags::Psi>(inbox_data)),
                              DataVector(face_size, psi_value));
        CHECK_ITERABLE_APPROX(
            get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(inbox_data)),
            DataVector(face_size, dt_psi_value));

        // ReceiveWorldtubeData
        CHECK(ActionTesting::next_action_if_ready<element_chare>(
            make_not_null(&runner), element_id));
        CHECK(element_inbox.empty());
        const auto& worldtube_solution =
            ActionTesting::get_databox_tag<element_chare,
                                           Tags::WorldtubeSolution<Dim>>(
                runner, element_id);
        CHECK_ITERABLE_APPROX(
            get(get<CurvedScalarWave::Tags::Psi>(worldtube_solution)),
            DataVector(face_size, psi_value));
        CHECK_ITERABLE_APPROX(
            get(get<CurvedScalarWave::Tags::Pi>(worldtube_solution)),
            DataVector(face_size, -dt_psi_value));
        for (size_t i = 0; i < Dim; ++i) {
          CHECK_ITERABLE_APPROX(
              get<CurvedScalarWave::Tags::Phi<Dim>>(worldtube_solution).get(i),
              DataVector(face_size, 0.));
        }
      } else {
        CHECK(element_inbox.empty());
      }
    }
  }
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
