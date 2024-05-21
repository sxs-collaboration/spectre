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
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/AnalyticChristoffel.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/FilterOptions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhRelativisticEuler/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean {
namespace {
double test(const size_t num_dg_pts) {
  using BoundaryCorrection = BoundaryCorrections::ProductOfCorrections<
      gh::BoundaryCorrections::UpwindPenalty<3>,
      ValenciaDivClean::BoundaryCorrections::Hll>;
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const Affine affine_map{-1.0, 1.0, -4.0, 4.0};
  const ElementId<3> element_id{
      0, {SegmentId{3, 4}, SegmentId{3, 4}, SegmentId{3, 4}}};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  Block<3> block{
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Affine3D{affine_map, affine_map, affine_map}),
      0,
      {}};
  ElementMap<3, Frame::Grid> element_map{
      element_id, block.is_time_dependent()
                      ? block.moving_mesh_logical_to_grid_map().get_clone()
                      : block.stationary_map().get_to_grid_frame()};
  const auto element = domain::Initialization::create_initial_element(
      element_id, block,
      std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{3, 3, 3}}});

  const auto moving_mesh_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});

  const gh::Solutions::WrappedGr<::RelativisticEuler::Solutions::TovStar> soln{
      1.28e-3,
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(100.0, 2.0)
          ->get_clone(),
      RelativisticEuler::Solutions::TovCoordinates::Schwarzschild};

  const double time = 0.0;
  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto dg_coords = moving_mesh_map(
      element_map(logical_coordinates(dg_mesh)), time, functions_of_time);

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
  for (const Direction<3>& direction : Direction<3>::all_directions()) {
    auto neighbor_logical_coords = logical_coordinates(subcell_mesh);
    neighbor_logical_coords.get(direction.dimension()) +=
        2.0 * direction.sign();
    auto neighbor_coords = moving_mesh_map(element_map(neighbor_logical_coords),
                                           time, functions_of_time);
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
        std::move(neighbor_data_in_direction);
  }

  DirectionMap<3, std::optional<Variables<
                      tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<3>>>>>
      normal_vectors{};
  for (const auto& direction : Direction<3>::all_directions()) {
    using inverse_spatial_metric_tag =
        typename System::inverse_spatial_metric_tag;
    const Mesh<2> face_mesh = dg_mesh.slice_away(direction.dimension());
    const auto face_logical_coords =
        interface_logical_coordinates(face_mesh, direction);
    std::unordered_map<Direction<3>, tnsr::i<DataVector, 3, Frame::Inertial>>
        unnormalized_normal_covectors{};
    tnsr::i<DataVector, 3, Frame::Inertial> unnormalized_covector{};
    const auto element_logical_to_grid_inv_jac =
        element_map.inv_jacobian(face_logical_coords);
    const auto grid_to_inertial_inv_jac = moving_mesh_map.inv_jacobian(
        element_map(face_logical_coords), time, functions_of_time);
    InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        element_logical_to_inertial_inv_jac{};
    for (size_t logical_i = 0; logical_i < 3; ++logical_i) {
      for (size_t inertial_i = 0; inertial_i < 3; ++inertial_i) {
        element_logical_to_inertial_inv_jac.get(logical_i, inertial_i) =
            element_logical_to_grid_inv_jac.get(logical_i, 0) *
            grid_to_inertial_inv_jac.get(0, inertial_i);
        for (size_t grid_i = 1; grid_i < 3; ++grid_i) {
          element_logical_to_inertial_inv_jac.get(logical_i, inertial_i) +=
              element_logical_to_grid_inv_jac.get(logical_i, grid_i) *
              grid_to_inertial_inv_jac.get(grid_i, inertial_i);
        }
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      unnormalized_covector.get(i) =
          element_logical_to_inertial_inv_jac.get(direction.dimension(), i);
    }
    unnormalized_normal_covectors[direction] = unnormalized_covector;
    Variables<tmpl::list<
        inverse_spatial_metric_tag,
        evolution::dg::Actions::detail::NormalVector<3>,
        evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>>
        fields_on_face{face_mesh.number_of_grid_points()};
    fields_on_face.assign_subset(
        soln.variables(moving_mesh_map(element_map(face_logical_coords), time,
                                       functions_of_time),
                       time, tmpl::list<inverse_spatial_metric_tag>{}));
    normal_vectors[direction] = std::nullopt;
    evolution::dg::Actions::detail::
        unit_normal_vector_and_covector_and_magnitude<System>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, moving_mesh_map);
  }

  using variables_tag = typename System::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  Variables<typename System::primitive_variables_tag::tags_list> dg_prim_vars{
      dg_mesh.number_of_grid_points()};
  dg_prim_vars.assign_subset(soln.variables(
      dg_coords, time, typename System::primitive_variables_tag::tags_list{}));
  typename variables_tag::type initial_variables{
      dg_mesh.number_of_grid_points()};
  initial_variables.assign_subset(soln.variables(
      dg_coords, time, typename System::gh_system::variables_tag::tags_list{}));

  const auto gamma1 =  // Gamma1, taken from SpEC BNS
      std::make_unique<
          gh::ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
          -0.999, 0.999 * 1.0,
          10.0 * 10.0,  // second 10 is "separation" of NSes
          std::array{0.0, 0.0, 0.0});
  const auto gamma2 =  // Gamma1, taken from SpEC BNS
      std::make_unique<
          gh::ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
          0.01, 1.35 * 1.0 / 1.4, 5.5 * 1.4, std::array{0.0, 0.0, 0.0});

  // Below are also dummy variables required for compilation due to boundary
  // condition FD ghost data. Since the element used here for testing has
  // neighbors in all directions, BoundaryConditionGhostData::apply() is not
  // actually called so it is okay to leave these variables somewhat poorly
  // initialized.
  std::optional<tnsr::I<DataVector, 3>> dummy_volume_mesh_velocity{};
  using DampingFunction =
      gh::ConstraintDamping::DampingFunction<3, Frame::Grid>;

  auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Element<3>, domain::Tags::Mesh<3>,
          evolution::dg::subcell::Tags::Mesh<3>, fd::Tags::Reconstructor,
          evolution::Tags::BoundaryCorrection<
              grmhd::GhValenciaDivClean::System>,
          hydro::Tags::GrmhdEquationOfState,
          typename System::primitive_variables_tag, dt_variables_tag,
          variables_tag,
          evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
          ValenciaDivClean::Tags::ConstraintDampingParameter,
          domain::Tags::ElementMap<3, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::MeshVelocity<3, Frame::Inertial>,
          evolution::dg::Tags::NormalCovectorAndMagnitude<3>, ::Tags::Time,
          domain::Tags::FunctionsOfTimeInitialize,
          gh::ConstraintDamping::Tags::DampingFunctionGamma0<3, Frame::Grid>,
          gh::ConstraintDamping::Tags::DampingFunctionGamma1<3, Frame::Grid>,
          gh::ConstraintDamping::Tags::DampingFunctionGamma2<3, Frame::Grid>,
          ::gh::gauges::Tags::GaugeCondition,
          grmhd::GhValenciaDivClean::fd::Tags::FilterOptions,
          evolution::dg::subcell::Tags::SubcellOptions<3>>,
      db::AddComputeTags<
          ::domain::Tags::LogicalCoordinates<3>,
          ::domain::Tags::MappedCoordinates<
              ::domain::Tags::ElementMap<3, Frame::Grid>,
              domain::Tags::Coordinates<3, Frame::ElementLogical>>,
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>,
          ::domain::Tags::MappedCoordinates<
              ::domain::Tags::ElementMap<3, Frame::Grid>,
              evolution::dg::subcell::Tags::Coordinates<3,
                                                        Frame::ElementLogical>,
              evolution::dg::subcell::Tags::Coordinates>,
          gr::Tags::SpatialMetricCompute<DataVector, 3, Frame::Inertial>,
          gr::Tags::DetAndInverseSpatialMetricCompute<DataVector, 3,
                                                      Frame::Inertial>,
          gr::Tags::SqrtDetSpatialMetricCompute<DataVector, 3, Frame::Inertial>,
          gh::ConstraintDamping::Tags::ConstraintGamma0Compute<3, Frame::Grid>,
          gh::ConstraintDamping::Tags::ConstraintGamma1Compute<3, Frame::Grid>,
          gh::ConstraintDamping::Tags::ConstraintGamma2Compute<3,
                                                               Frame::Grid>>>(
      element, dg_mesh, subcell_mesh,
      std::unique_ptr<grmhd::GhValenciaDivClean::fd::Reconstructor>{
          std::make_unique<
              grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim>()},
      std::unique_ptr<
          grmhd::GhValenciaDivClean::BoundaryCorrections::BoundaryCorrection>{
          std::make_unique<BoundaryCorrection>()},
      soln.equation_of_state().promote_to_3d_eos(), dg_prim_vars,
      // Set incorrect size for dt variables because they should get resized.
      Variables<typename dt_variables_tag::tags_list>{}, initial_variables,
      neighbor_data, 1.0, std::move(element_map), moving_mesh_map.get_clone(),
      dummy_volume_mesh_velocity, normal_vectors, time,
      clone_unique_ptrs(functions_of_time),
      // Note: These damping functions all assume Grid==Inertial. We need to
      // rescale the widths in the Grid frame for binaries.
      std::unique_ptr<DampingFunction>(  // Gamma0, taken from SpEC BNS
          std::make_unique<
              gh::ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
              0.01, 0.09 * 1.0 / 1.4, 5.5 * 1.4, std::array{0.0, 0.0, 0.0})),
      gamma1->get_clone(), gamma2->get_clone(),
      std::unique_ptr<gh::gauges::GaugeCondition>(
          std::make_unique<gh::gauges::AnalyticChristoffel>(soln.get_clone())),
      grmhd::GhValenciaDivClean::fd::FilterOptions{0.001},
      evolution::dg::subcell::SubcellOptions{
          4.0, 1_st, 1.0e-3, 1.0e-4, false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
          std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1});

  db::mutate_apply<ValenciaDivClean::ConservativeFromPrimitive>(
      make_not_null(&box));

  std::vector<DirectionalId<3>> mortars_to_reconstruct_to{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    mortars_to_reconstruct_to.emplace_back(
        DirectionalId<3>{direction, *neighbors.begin()});
  }

  const auto all_packaged_data =
      subcell::NeighborPackagedData::apply(box, mortars_to_reconstruct_to);

  // Parse out evolved vars, since those are easiest to check for correctness,
  // then return absolute difference between analytic and reconstructed values.
  DirectionalIdMap<3, typename variables_tag::type> evolved_vars_errors{};
  double max_rel_error = 0.0;
  for (const auto& [direction_and_id, data] : all_packaged_data) {
    const auto& direction = direction_and_id.direction();
    const Mesh<2> dg_interface_mesh = dg_mesh.slice_away(direction.dimension());

    using dg_package_field_tags =
        typename BoundaryCorrection::dg_package_field_tags;
    const Variables<dg_package_field_tags> packaged_data{
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<double*>(data.data()), data.size()};

    auto sliced_vars = data_on_slice(
        db::get<variables_tag>(box), dg_mesh.extents(), direction.dimension(),
        direction.side() == Side::Upper
            ? dg_mesh.extents(direction.dimension()) - 1
            : 0);

    tmpl::for_each<
        typename System::grmhd_system::variables_tag::type::tags_list>(
        [&sliced_vars, &max_rel_error, &packaged_data](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          auto& sliced_tensor = get<tag>(sliced_vars);
          const auto& packaged_data_tensor = get<tag>(packaged_data);
          for (size_t tensor_index = 0; tensor_index < sliced_tensor.size();
               ++tensor_index) {
            max_rel_error = std::max(
                max_rel_error,
                max(abs(sliced_tensor[tensor_index] -
                        packaged_data_tensor[tensor_index])) /
                    std::max({max(abs(sliced_tensor[tensor_index])),
                              max(abs(packaged_data_tensor[tensor_index])),
                              1.0e-17}));
          }
        });
    auto& sliced_spacetime_metric =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(sliced_vars);
    const auto& packaged_data_spacetime_metric =
        get<gh::Tags::VSpacetimeMetric<DataVector, 3>>(packaged_data);

    const auto spatial_metric = gr::spatial_metric(sliced_spacetime_metric);
    const auto [det_spatial_metric, inverse_spatial_metric] =
        determinant_and_inverse(spatial_metric);
    const auto shift =
        gr::shift(sliced_spacetime_metric, inverse_spatial_metric);
    const auto lapse = gr::lapse(shift, sliced_spacetime_metric);
    // Need normal vector...
    tnsr::i<DataVector, 3, Frame::Inertial> normal_covector(get(lapse).size(),
                                                            0.0);
    normal_covector.get(direction.dimension()) = direction.sign();
    const auto normal_magnitude =
        magnitude(normal_covector, inverse_spatial_metric);
    normal_covector.get(direction.dimension()) /= get(normal_magnitude);
    tnsr::I<DataVector, 3, Frame::Inertial> normal_vector(get(lapse).size(),
                                                          0.0);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        normal_vector.get(i) +=
            inverse_spatial_metric.get(i, j) * normal_covector.get(j);
      }
    }

    const auto dg_interface_grid_coords =
        db::get<domain::Tags::ElementMap<3, Frame::Grid>>(box)(
            interface_logical_coordinates(dg_interface_mesh, direction));
    Scalar<DataVector> gamma2_sdv{dg_interface_mesh.number_of_grid_points()};
    (*gamma2)(make_not_null(&gamma2_sdv), dg_interface_grid_coords, 0.0, {});

    const auto& pi = get<gh::Tags::Pi<DataVector, 3>>(sliced_vars);
    const auto& phi = get<gh::Tags::Phi<DataVector, 3>>(sliced_vars);
    tnsr::aa<DataVector, 3> v_plus_times_lambda_plus = pi;
    tnsr::aa<DataVector, 3> v_minus_times_lambda_minus = pi;
    for (size_t a = 0; a < 4; ++a) {
      for (size_t b = a; b < 4; ++b) {
        for (size_t i = 0; i < 3; ++i) {
          v_plus_times_lambda_plus.get(a, b) +=
              normal_vector.get(i) * phi.get(i, a, b);
          v_minus_times_lambda_minus.get(a, b) -=
              normal_vector.get(i) * phi.get(i, a, b);
        }
        v_plus_times_lambda_plus.get(a, b) -=
            get(gamma2_sdv) * sliced_spacetime_metric.get(a, b);
        v_minus_times_lambda_minus.get(a, b) -=
            get(gamma2_sdv) * sliced_spacetime_metric.get(a, b);

        // Multiply by char speed. Note that shift is zero in TOV
        v_plus_times_lambda_plus.get(a, b) *= get(lapse);
        v_minus_times_lambda_minus.get(a, b) *= -get(lapse);
      }
    }

    const auto& packaged_data_v_plus_times_lambda_plus =
        get<gh::Tags::VPlus<DataVector, 3>>(packaged_data);
    const auto& packaged_data_v_minus_times_lambda_minus =
        get<gh::Tags::VMinus<DataVector, 3>>(packaged_data);
    for (size_t tensor_index = 0; tensor_index < sliced_spacetime_metric.size();
         ++tensor_index) {
      max_rel_error = std::max(
          max_rel_error,
          max(abs(
              // Note: this is first char speed * g_{ab}
              get<tmpl::at_c<dg_package_field_tags, 7>>(packaged_data)[0] *
                  sliced_spacetime_metric[tensor_index] -
              packaged_data_spacetime_metric[tensor_index])) /
              std::max({max(abs(sliced_spacetime_metric[tensor_index])),
                        max(abs(packaged_data_spacetime_metric[tensor_index])),
                        1.0e-17}));
      max_rel_error = std::max(
          max_rel_error,
          max(abs((direction.side() == Side::Upper ? -1.0 : 1.0) *
                      (direction.side() == Side::Upper
                           ? v_minus_times_lambda_minus[tensor_index]
                           : v_plus_times_lambda_plus[tensor_index]) -
                  packaged_data_v_plus_times_lambda_plus[tensor_index])) /
              std::max(
                  {max(abs((direction.side() == Side::Upper
                                ? v_minus_times_lambda_minus[tensor_index]
                                : v_plus_times_lambda_plus[tensor_index]))),
                   max(abs(
                       packaged_data_v_plus_times_lambda_plus[tensor_index])),
                   1.0e-17}));
      max_rel_error = std::max(
          max_rel_error,
          max(abs((direction.side() == Side::Lower ? 1.0 : -1.0) *
                      (direction.side() == Side::Lower
                           ? v_minus_times_lambda_minus[tensor_index]
                           : v_plus_times_lambda_plus[tensor_index]) -
                  packaged_data_v_minus_times_lambda_minus[tensor_index])) /
              std::max(
                  {max(abs((direction.side() == Side::Lower
                                ? v_minus_times_lambda_minus[tensor_index]
                                : v_plus_times_lambda_plus[tensor_index]))),
                   max(abs(
                       packaged_data_v_minus_times_lambda_minus[tensor_index])),
                   1.0e-17}));
    }
  }
  return max_rel_error;
}

// [[Timeout, 10]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GhValenciaDivClean.Subcell.NeighborPackagedData",
    "[Unit][Evolution]") {
  // This tests sets up a cube [-4,4]^3 in a TOV star spacetime and verifies
  // that the time derivative vanishes. Or, more specifically, that the time
  // derivative decreases with increasing resolution and is below 1.0e-4.
  const double error_4 = test(4);
  const double error_8 = test(8);
  CHECK(error_4 > error_8);
  // Check that the error is "reasonably small"
  CHECK(error_8 < 1.0e-4);
}
}  // namespace
}  // namespace grmhd::GhValenciaDivClean
