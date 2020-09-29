// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/Systems/ScalarWave/Characteristics.hpp"
#include "Evolution/Systems/ScalarWave/Constraints.hpp"
#include "Evolution/Systems/ScalarWave/Equations.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Evolution/Systems/ScalarWave/TimeDerivative.hpp"
#include "Evolution/Systems/ScalarWave/UpwindPenaltyCorrection.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SemidiscretizedDg.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<1>;

  using variables_tag = typename metavariables::system::variables_tag;
  using boundary_scheme = typename metavariables::boundary_scheme;
  using normal_dot_fluxes_tag = domain::Tags::Interface<
      domain::Tags::InternalDirections<1>,
      db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>>;
  using mortar_data_tag =
      Tags::Mortars<typename boundary_scheme::mortar_data_tag, 1>;

  using const_global_cache_tags =
      tmpl::list<typename boundary_scheme::numerical_flux_computer_tag>;

  using simple_tags = db::AddSimpleTags<
      Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, domain::Tags::Mesh<1>,
      domain::Tags::Element<1>, domain::Tags::MeshVelocity<1>,
      domain::Tags::DivMeshVelocity, domain::Tags::ElementMap<1>, variables_tag,
      db::add_tag_prefix<Tags::dt, variables_tag>,
      ScalarWave::Tags::ConstraintGamma2, normal_dot_fluxes_tag,
      mortar_data_tag, Tags::Mortars<Tags::Next<Tags::TimeStepId>, 1>,
      Tags::Mortars<domain::Tags::Mesh<0>, 1>,
      Tags::Mortars<Tags::MortarSize<0>, 1>,
      // Need this only because the DG scheme doesn't know at compile-time that
      // the element has no external boundaries
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<1>,
          Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<1>>>>;

  using inverse_jacobian = domain::Tags::InverseJacobianCompute<
      domain::Tags::ElementMap<1>,
      domain::Tags::Coordinates<1, Frame::Logical>>;

  template <typename Tag>
  using interface_compute_tag =
      domain::Tags::InterfaceCompute<domain::Tags::InternalDirections<1>, Tag>;

  using compute_tags = db::AddComputeTags<
      domain::Tags::LogicalCoordinates<1>,
      domain::Tags::MappedCoordinates<
          domain::Tags::ElementMap<1>,
          domain::Tags::Coordinates<1, Frame::Logical>>,
      inverse_jacobian,
      Tags::DerivCompute<variables_tag, inverse_jacobian,
                         typename metavariables::system::gradients_tags>,
      domain::Tags::InternalDirectionsCompute<1>,
      domain::Tags::Slice<domain::Tags::InternalDirections<1>,
                          typename metavariables::system::variables_tag>,
      domain::Tags::Slice<domain::Tags::InternalDirections<1>,
                          ScalarWave::Tags::ConstraintGamma2>,
      interface_compute_tag<domain::Tags::Direction<1>>,
      interface_compute_tag<domain::Tags::InterfaceMesh<1>>,
      interface_compute_tag<domain::Tags::UnnormalizedFaceNormalCompute<1>>,
      interface_compute_tag<
          Tags::EuclideanMagnitude<domain::Tags::UnnormalizedFaceNormal<1>>>,
      interface_compute_tag<
          Tags::NormalizedCompute<domain::Tags::UnnormalizedFaceNormal<1>>>,
      interface_compute_tag<ScalarWave::Tags::CharacteristicFieldsCompute<1>>,
      interface_compute_tag<ScalarWave::Tags::CharacteristicSpeedsCompute<1>>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              evolution::dg::Actions::ComputeTimeDerivative<Metavariables>,
              dg::Actions::SendDataForFluxes<boundary_scheme>,
              dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
              Actions::MutateApply<boundary_scheme>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;
  using system = ScalarWave::System<1>;
  using boundary_scheme = dg::FirstOrderScheme::FirstOrderScheme<
      1, typename system::variables_tag,
      db::add_tag_prefix<::Tags::dt, typename system::variables_tag>,
      Tags::NumericalFlux<ScalarWave::UpwindPenaltyCorrection<1>>,
      Tags::TimeStepId>;

  using component_list = tmpl::list<Component<Metavariables>>;
  using temporal_id = Tags::TimeStepId;
  enum class Phase { Initialization, Testing, Exit };
};

using system = Metavariables::system;
using EvolvedVariables = typename system::variables_tag::type;

std::pair<tnsr::I<DataVector, 1>, EvolvedVariables> evaluate_rhs(
    const double time,
    const ScalarWave::Solutions::SemidiscretizedDg& solution) noexcept {
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {ScalarWave::UpwindPenaltyCorrection<1>{}}};

  const Slab slab(time, time + 1.0);
  const TimeStepId current_time(true, 0, slab.start());
  const TimeStepId next_time(true, 0, slab.end());
  const Mesh<1> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                       domain::CoordinateMaps::Affine>));
  const auto block_map =
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::Affine(-1.0, 1.0, 0.0, 2.0 * M_PI));

  const auto emplace_element =
      [&block_map, &current_time, &mesh, &next_time, &runner, &solution, &time](
          const ElementId<1>& id,
          const std::vector<std::pair<Direction<1>, ElementId<1>>>&
              mortars) noexcept {
        Element<1>::Neighbors_t neighbors;
        for (const auto& mortar_id : mortars) {
          neighbors.insert({mortar_id.first, {{mortar_id.second}, {}}});
        }
        const Element<1> element(id, std::move(neighbors));

        auto map = ElementMap<1, Frame::Inertial>(id, block_map->get_clone());

        typename system::variables_tag::type variables(2);
        typename db::add_tag_prefix<Tags::dt, system::variables_tag>::type
            dt_variables(2);

        typename component::normal_dot_fluxes_tag::type normal_dot_fluxes;
        typename component::mortar_data_tag::type mortar_history{};
        typename Tags::Mortars<Tags::Next<Tags::TimeStepId>, 1>::type
            mortar_next_temporal_ids{};
        typename Tags::Mortars<domain::Tags::Mesh<0>, 1>::type mortar_meshes{};
        typename Tags::Mortars<Tags::MortarSize<0>, 1>::type mortar_sizes{};
        for (const auto& mortar_id : mortars) {
          normal_dot_fluxes[mortar_id.first].initialize(1, 0.0);
          mortar_history.insert({mortar_id, {}});
          mortar_next_temporal_ids.insert({mortar_id, current_time});
          mortar_meshes.insert({mortar_id, mesh.slice_away(0)});
          mortar_sizes.insert({mortar_id, {}});
        }
        Scalar<DataVector> gamma_2{mesh.number_of_grid_points(), 0.};

        ActionTesting::emplace_component_and_initialize<component>(
            &runner, id,
            {current_time, next_time, mesh, element,
             boost::optional<tnsr::I<DataVector, 1, Frame::Inertial>>{},
             boost::optional<Scalar<DataVector>>{}, std::move(map),
             std::move(variables), std::move(dt_variables), std::move(gamma_2),
             std::move(normal_dot_fluxes), std::move(mortar_history),
             std::move(mortar_next_temporal_ids), std::move(mortar_meshes),
             std::move(mortar_sizes),
             std::unordered_map<Direction<1>, Scalar<DataVector>>{}});

        auto& box = ActionTesting::get_databox<
            component,
            tmpl::append<component::simple_tags, component::compute_tags>>(
            make_not_null(&runner), id);
        db::mutate<system::variables_tag>(
            make_not_null(&box),
            [
              &solution, &time
            ](const gsl::not_null<EvolvedVariables*> vars,
              const tnsr::I<DataVector, 1, Frame::Inertial>& coords) noexcept {
              vars->assign_subset(solution.variables(
                  coords, time, system::variables_tag::tags_list{}));
            },
            db::get<domain::Tags::Coordinates<1, Frame::Inertial>>(box));
      };

  const ElementId<1> self_id(0, {{{4, 1}}});
  const ElementId<1> left_id(0, {{{4, 0}}});
  const ElementId<1> right_id(0, {{{4, 2}}});

  emplace_element(self_id, {{Direction<1>::lower_xi(), left_id},
                            {Direction<1>::upper_xi(), right_id}});
  emplace_element(left_id, {{Direction<1>::upper_xi(), self_id}});
  emplace_element(right_id, {{Direction<1>::lower_xi(), self_id}});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  // The neighbors only have to get as far as sending data
  for (size_t i = 0; i < 2; ++i) {
    runner.next_action<component>(left_id);
    runner.next_action<component>(right_id);
  }

  for (size_t i = 0; i < 4; ++i) {
    runner.next_action<component>(self_id);
  }

  return {
      ActionTesting::get_databox_tag<
          component, domain::Tags::Coordinates<1, Frame::Inertial>>(runner,
                                                                    self_id),
      EvolvedVariables(
          ActionTesting::get_databox_tag<
              component, db::add_tag_prefix<Tags::dt, system::variables_tag>>(
              runner, self_id))};
}

void check_solution(
    const ScalarWave::Solutions::SemidiscretizedDg& solution) noexcept {
  const double time = 1.23;

  const auto coords_and_deriv_from_system = evaluate_rhs(time, solution);
  const auto& coords = coords_and_deriv_from_system.first;
  const auto& deriv_from_system = coords_and_deriv_from_system.second;
  const auto numerical_deriv = numerical_derivative(
      [&coords, &solution](const std::array<double, 1>& t) noexcept {
        EvolvedVariables result(2);
        result.assign_subset(solution.variables(
            coords, t[0], system::variables_tag::tags_list{}));
        return result;
      },
      std::array<double, 1>{{time}}, 0, 1e-3);

  CHECK_VARIABLES_CUSTOM_APPROX(numerical_deriv, deriv_from_system,
                                approx.epsilon(1.0e-12));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.WaveEquation.SemidiscretizedDg",
    "[Unit][PointwiseFunctions]") {
  // We use 16 elements, so there are 16 independent harmonics.
  for (int harmonic = 0; harmonic < 16; ++harmonic) {
    check_solution({harmonic, {{1.2, 2.3, 3.4, 4.5}}});
  }

  check_solution(
      TestHelpers::test_creation<ScalarWave::Solutions::SemidiscretizedDg>(
          "Harmonic: 1\n"
          "Amplitudes: [1.2, 2.3, 3.4, 4.5]"));
}
