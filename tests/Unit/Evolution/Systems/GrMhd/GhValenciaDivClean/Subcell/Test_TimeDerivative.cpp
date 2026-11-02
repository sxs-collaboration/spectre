// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/AnalyticChristoffel.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/FilterOptions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhRelativisticEuler/Factory.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace grmhd::GhValenciaDivClean {
namespace {

// These solution tag and metavariables are not strictly required for testing
// subcell time derivative, but needed for compilation since
// BoundaryConditionGhostData requires this to be in box.
struct DummyAnalyticSolutionTag : db::SimpleTag {
  using type =
      gh::Solutions::WrappedGr<::RelativisticEuler::Solutions::TovStar>;
};

template <typename System>
struct DummyEvolutionMetaVars {
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = true;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<BoundaryConditions::BoundaryCondition,
                   tmpl::push_back<
                       BoundaryConditions::standard_boundary_conditions<System>,
                       BoundaryConditions::DirichletAnalytic<System>>>,
        tmpl::pair<evolution::BoundaryCorrection,
                   BoundaryCorrections::standard_boundary_corrections>,
        tmpl::pair<evolution::initial_data::InitialData,
                   ghmhd::GhValenciaDivClean::InitialData::
                       analytic_solutions_and_data_list>>;
  };
};

template <typename System>
double test(const size_t num_dg_pts, std::optional<double> expansion_velocity,
            const bool test_non_diagonal_jacobian) {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  const gh::Solutions::WrappedGr<::RelativisticEuler::Solutions::TovStar> soln{
      1.28e-3,
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(100.0, 2.0)
          ->get_clone(),
      RelativisticEuler::Solutions::TovCoordinates::Schwarzschild};
  const Affine affine_map{-1.0, 1.0, -4.0, 4.0};
  const ElementId<3> element_id{
      0, {SegmentId{3, 4}, SegmentId{3, 4}, SegmentId{3, 7}}};
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_boundary_conditions(1);
  for (const auto& direction : Direction<3>::all_directions()) {
    external_boundary_conditions.at(0)[direction] =
        grmhd::GhValenciaDivClean::BoundaryConditions::DirichletAnalytic<
            System>(std::make_unique<gh::Solutions::WrappedGr<
                        ::RelativisticEuler::Solutions::TovStar>>(soln))
            .get_clone();
  }
  std::vector<Block<3>> blocks;
  blocks.emplace_back(
      Block<3>{test_non_diagonal_jacobian
                   ? domain::make_coordinate_map_base<Frame::BlockLogical,
                                                      Frame::Inertial>(
                         ::domain::CoordinateMaps::Rotation<3>(0.7, 0, 0.))
                   : domain::make_coordinate_map_base<Frame::BlockLogical,
                                                      Frame::Inertial>(
                         Affine3D{affine_map, affine_map, affine_map}),
               0,
               {}});
  const auto& block = blocks[0];

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  ElementMap<3, Frame::Grid> element_map{
      element_id, block.is_time_dependent()
                      ? block.moving_mesh_logical_to_grid_map().get_clone()
                      : block.stationary_map().get_to_grid_frame()};
  const auto grid_to_inertial_map =
      ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          ::domain::CoordinateMaps::Identity<3>{});
  const auto element = domain::create_initial_element(
      element_id, blocks,
      std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{3, 3, 3}}});

  const double time = 0.0;
  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t num_dg_pts_3d = num_dg_pts * num_dg_pts * num_dg_pts;

  const auto logical_coords = logical_coordinates(subcell_mesh);
  const auto cell_centered_coords = (*grid_to_inertial_map)(
      element_map(logical_coords), time, functions_of_time);
  const auto dg_coords = (*grid_to_inertial_map)(
      element_map(logical_coordinates(dg_mesh)), time, functions_of_time);

  const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>
      cell_centered_logical_to_grid_inv_jacobian =
          element_map.inv_jacobian(logical_coords);
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian =
          InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>(
              subcell_mesh.number_of_grid_points());
  const auto& cell_centered_grid_to_inertial_inv_jacobian =
      grid_to_inertial_map->inv_jacobian(element_map(logical_coords));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      auto& inv_jacobian_component =
          cell_centered_logical_to_inertial_inv_jacobian.get(i, j);
      inv_jacobian_component = 0.;
      for (size_t k = 0; k < 3; k++) {
        inv_jacobian_component +=
            cell_centered_logical_to_grid_inv_jacobian.get(i, k) *
            cell_centered_grid_to_inertial_inv_jacobian.get(k, j);
      }
    }
  }

  const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>
      dg_logical_to_grid_inv_jacobian =
          element_map.inv_jacobian(logical_coordinates(dg_mesh));
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      dg_logical_to_inertial_inv_jacobian =
          InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>(dg_mesh.number_of_grid_points());
  const auto& dg_grid_to_inertial_inv_jacobian =
      grid_to_inertial_map->inv_jacobian(
          element_map(logical_coordinates(dg_mesh)));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      auto& inv_jacobian_component =
          dg_logical_to_inertial_inv_jacobian.get(i, j);
      inv_jacobian_component = 0.;
      for (size_t k = 0; k < 3; k++) {
        inv_jacobian_component += dg_logical_to_grid_inv_jacobian.get(i, k) *
                                  dg_grid_to_inertial_inv_jacobian.get(k, j);
      }
    }
  }

  using variables_tag = typename System::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using evolved_tags = typename System::variables_tag::tags_list;
  using conserved_tags =
      typename grmhd::ValenciaDivClean::ConservativeFromPrimitive::return_tags;
  using gh_variables_tag = typename System::gh_system::variables_tag;
  using gh_variables_tags = typename gh_variables_tag::tags_list;
  Variables<typename System::primitive_variables_tag::tags_list>
      cell_centered_prim_vars{subcell_mesh.number_of_grid_points()};
  cell_centered_prim_vars.assign_subset(
      soln.variables(cell_centered_coords, time,
                     typename System::primitive_variables_tag::tags_list{}));
  typename variables_tag::type initial_variables{
      subcell_mesh.number_of_grid_points()};
  initial_variables.assign_subset(
      soln.variables(cell_centered_coords, time,
                     typename System::gh_system::variables_tag::tags_list{}));
  std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> dg_mesh_velocity{};
  std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>
      subcell_mesh_velocity{};
  if (expansion_velocity.has_value()) {
    dg_mesh_velocity = std::optional<tnsr::I<DataVector, 3>>(
        tnsr::I<DataVector, 3>(num_dg_pts_3d));
    for (int i = 0; i < 3; i++) {
      dg_mesh_velocity.value().get(i) =
          dg_coords.get(i) * expansion_velocity.value();
    }

    subcell_mesh_velocity = std::optional<tnsr::I<DataVector, 3>>(
        tnsr::I<DataVector, 3>(subcell_mesh.number_of_grid_points()));
    for (int i = 0; i < 3; i++) {
      subcell_mesh_velocity.value().get(i) =
          cell_centered_coords.get(i) * expansion_velocity.value();
    }
  }
  std::optional<Scalar<DataVector>> div_dg_mesh_velocity{};
  if (expansion_velocity.has_value()) {
    div_dg_mesh_velocity =
        std::optional<Scalar<DataVector>>(Scalar<DataVector>(num_dg_pts_3d));
    div_dg_mesh_velocity.value().get() = 3.0 * expansion_velocity.value();
  }
  // To be recomputed on each face.
  std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};

  // Neighbor data for reconstruction.
  //
  // 0. neighbors coords (our logical coords +2)
  // 1. compute prims from solution
  // 2. compute prims needed for reconstruction
  // 3. set neighbor data
  evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      neighbor_data{};
  using prims_to_reconstruct_tags = grmhd::GhValenciaDivClean::Tags::
      primitive_grmhd_and_spacetime_reconstruction_tags;
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    auto neighbor_logical_coords = logical_coordinates(subcell_mesh);
    neighbor_logical_coords.get(direction.dimension()) +=
        2.0 * direction.sign();
    auto neighbor_coords = (*grid_to_inertial_map)(
        element_map(neighbor_logical_coords), time, functions_of_time);
    const auto neighbor_prims = soln.variables(
        neighbor_coords, time,
        tmpl::append<typename System::primitive_variables_tag::tags_list,
                     typename System::gh_system::variables_tag::tags_list>{});
    Variables<prims_to_reconstruct_tags> prims_to_reconstruct{
        subcell_mesh.number_of_grid_points()};
    prims_to_reconstruct.assign_subset(neighbor_prims);
    get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
        prims_to_reconstruct) =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(neighbor_prims);
    for (auto& component :
         get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
             prims_to_reconstruct)) {
      component *=
          get(get<hydro::Tags::LorentzFactor<DataVector>>(neighbor_prims));
    }

    // Slice data so we can add it to the element's neighbor data
    DataVector neighbor_data_in_direction =
        evolution::dg::subcell::slice_data(
            prims_to_reconstruct, subcell_mesh.extents(),
            grmhd::ValenciaDivClean::fd::MonotonisedCentralPrim{}
                .ghost_zone_size(),
            std::unordered_set{direction.opposite()}, 0, {})
            .at(direction.opposite());
    const auto key =
        DirectionalId<3>{direction, *element.neighbors().at(direction).begin()};
    neighbor_data[key] = evolution::dg::subcell::GhostData{1};
    neighbor_data[key].neighbor_ghost_data_for_reconstruction() =
        neighbor_data_in_direction;
  }

  Domain<3> domain{std::move(blocks)};

  const auto gamma1 =  // Gamma1, taken from SpEC BNS
      std::make_unique<ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
          -0.999, 0.999 * 1.0,
          10.0 * 10.0,  // second 10 is "separation" of NSes
          std::array{0.0, 0.0, 0.0});
  const auto gamma2 =  // Gamma2, taken from SpEC BNS
      std::make_unique<ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
          0.01, 1.35 * 1.0 / 1.4, 5.5 * 1.4, std::array{0.0, 0.0, 0.0});

  evolution::dg::Tags::MortarData<3>::type mortar_data{};

  // Below are also dummy variables required for compilation due to boundary
  // condition FD ghost data. Since the element used here for testing has
  // neighbors in all directions, BoundaryConditionGhostData::apply() is not
  // actually called so it is okay to leave these variables somewhat poorly
  // initialized.
  typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type
      dummy_normal_covector_and_magnitude{};
  using DampingFunction = ConstraintDamping::DampingFunction<3, Frame::Grid>;
  typename evolution::dg::subcell::Tags::ReconstructionOrder<3>::type
      dummy_reconstruction_order{};

  auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Element<3>, evolution::dg::subcell::Tags::Mesh<3>,
          domain::Tags::Mesh<3>, fd::Tags::Reconstructor<System>,
          evolution::Tags::BoundaryCorrection,
          hydro::Tags::GrmhdEquationOfState,
          typename System::primitive_variables_tag, dt_variables_tag,
          variables_tag,
          evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
          evolution::dg::subcell::Tags::ReconstructionOrder<3>,
          ValenciaDivClean::Tags::ConstraintDampingParameter,
          evolution::dg::Tags::MortarData<3>,
          domain::Tags::ElementMap<3, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::Domain<3>, domain::Tags::ExternalBoundaryConditions<3>,
          domain::Tags::MeshVelocity<3, Frame::Inertial>,
          domain::Tags::DivMeshVelocity,
          domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                        Frame::Inertial>,
          evolution::dg::Tags::NormalCovectorAndMagnitude<3>, ::Tags::Time,
          domain::Tags::FunctionsOfTimeInitialize,
          Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<System>>,
          gh::Tags::DampingFunctionGamma0<3, Frame::Grid>,
          gh::Tags::DampingFunctionGamma1<3, Frame::Grid>,
          gh::Tags::DampingFunctionGamma2<3, Frame::Grid>,
          ::gh::gauges::Tags::GaugeCondition,
          grmhd::GhValenciaDivClean::fd::Tags::FilterOptions,
          ::Tags::VariableFixer<::VariableFixing::FixToAtmosphere<3>>>,
      db::AddComputeTags<
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>,
          ::domain::Tags::MappedCoordinates<
              ::domain::Tags::ElementMap<3, Frame::Grid>,
              evolution::dg::subcell::Tags::Coordinates<3,
                                                        Frame::ElementLogical>,
              evolution::dg::subcell::Tags::Coordinates>,
          evolution::dg::subcell::Tags::InertialCoordinatesCompute<
              ::domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                            Frame::Inertial>>,
          evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGridCompute<
              ::domain::Tags::ElementMap<3, Frame::Grid>, 3>,
          evolution::dg::subcell::fd::Tags::
              DetInverseJacobianLogicalToGridCompute<3>,
          evolution::dg::subcell::fd::Tags::
              InverseJacobianLogicalToInertialCompute<
                  ::domain::CoordinateMaps::Tags::CoordinateMap<
                      3, Frame::Grid, Frame::Inertial>,
                  3>,
          evolution::dg::subcell::fd::Tags::
              DetInverseJacobianLogicalToInertialCompute<
                  ::domain::CoordinateMaps::Tags::CoordinateMap<
                      3, Frame::Grid, Frame::Inertial>,
                  3>,
          domain::Tags::DetInvJacobianCompute<3, Frame::ElementLogical,
                                              Frame::Inertial>,
          gr::Tags::SpatialMetricCompute<DataVector, 3, Frame::Inertial>,
          gr::Tags::DetAndInverseSpatialMetricCompute<DataVector, 3,
                                                      Frame::Inertial>,
          gr::Tags::SqrtDetSpatialMetricCompute<DataVector, 3,
                                                Frame::Inertial>>>(
      element, subcell_mesh, dg_mesh,
      std::unique_ptr<grmhd::GhValenciaDivClean::fd::Reconstructor<System>>{
          std::make_unique<
              grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<System>>()},
      std::unique_ptr<evolution::BoundaryCorrection>{
          std::make_unique<BoundaryCorrections::ProductOfCorrections<
              gh::BoundaryCorrections::UpwindPenalty<3>,
              ValenciaDivClean::BoundaryCorrections::Hll>>(
              gh::BoundaryCorrections::UpwindPenalty<3>{},
              ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8})},
      soln.equation_of_state().promote_to_3d_eos(), cell_centered_prim_vars,
      // Set incorrect size for dt variables because they should get resized.
      Variables<typename dt_variables_tag::tags_list>{}, initial_variables,
      neighbor_data, dummy_reconstruction_order, 1.0, mortar_data,
      std::move(element_map),
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}),
      std::move(domain), std::move(external_boundary_conditions),
      dg_mesh_velocity, div_dg_mesh_velocity,
      dg_logical_to_inertial_inv_jacobian, dummy_normal_covector_and_magnitude,
      time, clone_unique_ptrs(functions_of_time),
      DummyEvolutionMetaVars<System>{},
      // Note: These damping functions all assume Grid==Inertial. We need to
      // rescale the widths in the Grid frame for binaries.
      std::unique_ptr<DampingFunction>(  // Gamma0, taken from SpEC BNS
          std::make_unique<
              ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
              0.01, 0.09 * 1.0 / 1.4, 5.5 * 1.4, std::array{0.0, 0.0, 0.0})),
      gamma1->get_clone(), gamma2->get_clone(),
      std::unique_ptr<gh::gauges::GaugeCondition>(
          std::make_unique<gh::gauges::AnalyticChristoffel>(soln.get_clone())),
      grmhd::GhValenciaDivClean::fd::FilterOptions{0.001},
      // Just use a default-constructed fixer and have the reconstructor not
      // call it.
      ::VariableFixing::FixToAtmosphere<3>{});

  db::mutate_apply<ValenciaDivClean::ConservativeFromPrimitive>(
      make_not_null(&box));

  subcell::TimeDerivative<System>::apply(make_not_null(&box));

  // We test that the time derivative converges to zero,
  // so we remove the expected value of the time derivative for moving meshes
  Variables<evolved_tags> output_minus_expected_dt_vars{
      subcell_mesh.number_of_grid_points()};
  const auto& dt_vars = db::get<dt_variables_tag>(box);

  tmpl::for_each<conserved_tags>(
      [&box, &subcell_mesh, &expansion_velocity, &cell_centered_coords,
       &cell_centered_logical_to_inertial_inv_jacobian,
       &output_minus_expected_dt_vars, &dt_vars](auto var_tag_v) {
        using var_tag = tmpl::type_from<decltype(var_tag_v)>;
        const auto& var = get<var_tag>(box);
        const auto deriv_var = partial_derivative(
            var, subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);

        auto& output_minus_expected_dt_var =
            get<var_tag>(output_minus_expected_dt_vars);
        const auto& output_dt_var = get<::Tags::dt<var_tag>>(dt_vars);
        for (size_t i = 0; i < output_minus_expected_dt_var.size(); ++i) {
          output_minus_expected_dt_var[i] = output_dt_var[i];
          if (expansion_velocity.has_value()) {
            for (size_t j = 0; j < 3; ++j) {
              const auto deriv_index =
                  j * output_minus_expected_dt_var.size() + i;
              output_minus_expected_dt_var[i] -= cell_centered_coords.get(j) *
                                                 deriv_var[deriv_index] *
                                                 expansion_velocity.value();
            }
          }
        }
      });

  // We need the gradient of the GR variable (and partial_derivative, as is,
  // does not work)
  using gh_gradient_tags = typename TimeDerivativeTerms::gh_gradient_tags;
  const auto& gh_evolved_vars = db::get<variables_tag>(box);
  Variables<db::wrap_tags_in<::Tags::deriv, gh_gradient_tags, tmpl::size_t<3>,
                             Frame::Inertial>>
      cell_centered_gh_derivs{subcell_mesh.number_of_grid_points()};
  const size_t fd_deriv_order = 4;
  grmhd::GhValenciaDivClean::fd::spacetime_derivatives<System>(
      make_not_null(&cell_centered_gh_derivs), gh_evolved_vars,
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box),
      fd_deriv_order, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian, dg_coords);

  auto& temp = get<gr::Tags::SpacetimeMetric<DataVector, 3>>(
      output_minus_expected_dt_vars);
  temp = get<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(dt_vars);

  tmpl::for_each<gh_variables_tags>([&expansion_velocity, &cell_centered_coords,
                                     &output_minus_expected_dt_vars, &dt_vars,
                                     &cell_centered_gh_derivs](auto var_tag_v) {
    using var_tag = tmpl::type_from<decltype(var_tag_v)>;
    using grad_tag = ::Tags::deriv<var_tag, tmpl::size_t<3>, Frame::Inertial>;
    using FluxTensor = typename grad_tag::type;
    const auto& deriv_var = get<grad_tag>(cell_centered_gh_derivs);

    auto& output_minus_expected_dt_var =
        get<var_tag>(output_minus_expected_dt_vars);
    const auto& output_dt_var = get<::Tags::dt<var_tag>>(dt_vars);
    for (size_t i = 0; i < output_minus_expected_dt_var.size(); ++i) {
      output_minus_expected_dt_var[i] = output_dt_var[i];
      if (expansion_velocity.has_value()) {
        const auto tensor_index = output_dt_var.get_tensor_index(i);
        for (size_t j = 0; j < 3; ++j) {
          const auto deriv_index =
              FluxTensor::get_storage_index(prepend(tensor_index, j));
          output_minus_expected_dt_var[i] -= cell_centered_coords.get(j) *
                                             deriv_var[deriv_index] *
                                             expansion_velocity.value();
        }
      }
    }
  });

  return std::max(
      {max(abs(get(get<grmhd::ValenciaDivClean::Tags::TildeD>(
           output_minus_expected_dt_vars)))),
       max(abs(get(get<grmhd::ValenciaDivClean::Tags::TildeYe>(
           output_minus_expected_dt_vars)))),
       max(abs(get(get<grmhd::ValenciaDivClean::Tags::TildeTau>(
           output_minus_expected_dt_vars)))),
       max(get(magnitude(get<grmhd::ValenciaDivClean::Tags::TildeS<>>(
           output_minus_expected_dt_vars)))),
       max(get(magnitude(get<grmhd::ValenciaDivClean::Tags::TildeB<>>(
           output_minus_expected_dt_vars)))),
       max(get(pointwise_l2_norm(get<gr::Tags::SpacetimeMetric<DataVector, 3>>(
           output_minus_expected_dt_vars)))),
       max(get(
           pointwise_l2_norm(get<gh::Tags::Pi<DataVector, 3, Frame::Inertial>>(
               output_minus_expected_dt_vars)))),
       max(get(
           pointwise_l2_norm(get<gh::Tags::Phi<DataVector, 3, Frame::Inertial>>(
               output_minus_expected_dt_vars))))});
}

// [[Timeout, 20]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GhValenciaDivClean.Subcell.TimeDerivative",
    "[Unit][Evolution]") {
  // This tests sets up a cube [-4,4]^3 in a TOV star spacetime and verifies
  // that the time derivative vanishes. Or, more specifically, that the time
  // derivative decreases with increasing resolution and is below 1.0e-6.

  using NeutrinoTransportSystem = RadiationTransport::NoNeutrinos::System;
  using System = grmhd::GhValenciaDivClean::System<NeutrinoTransportSystem>;

  std::optional<double> expansion_velocity = {};

  CHECK(test<System>(4, expansion_velocity, false) >
        test<System>(8, expansion_velocity, false));
  CHECK(test<System>(8, expansion_velocity, false) < 1.0e-6);

  expansion_velocity = 0.5;

  CHECK(test<System>(4, expansion_velocity, false) >
        test<System>(8, expansion_velocity, false));
  CHECK(test<System>(8, expansion_velocity, false) < 1.0e-6);

  CHECK(test<System>(4, expansion_velocity, true) >
        test<System>(8, expansion_velocity, true));
  CHECK(test<System>(8, expansion_velocity, true) < 1.0e-6);
}
}  // namespace
}  // namespace grmhd::GhValenciaDivClean
