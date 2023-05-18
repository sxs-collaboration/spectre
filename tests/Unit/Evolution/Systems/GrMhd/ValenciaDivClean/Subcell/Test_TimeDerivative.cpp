// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace grmhd::ValenciaDivClean {
namespace {

// These solution tag and metavariables are not strictly required for testing
// subcell time derivative, but needed for compilation since
// BoundaryConditionGhostData requires this to be in box.
struct DummyAnalyticSolutionTag : db::SimpleTag,
                                  ::Tags::AnalyticSolutionOrData {
  using type = Solutions::SmoothFlow;
};

struct DummyEvolutionMetaVars {
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = false;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BoundaryConditions::BoundaryCondition,
                             BoundaryConditions::standard_boundary_conditions>>;
  };
};

template <size_t Dim, typename... Maps, typename Solution>
auto face_centered_gr_tags(
    const Mesh<Dim>& subcell_mesh, const double time,
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                Maps...>& map,
    const Solution& soln) {
  std::array<typename System::flux_spacetime_variables_tag::type, Dim>
      face_centered_gr_vars{};

  for (size_t d = 0; d < Dim; ++d) {
    const auto basis = make_array<Dim>(subcell_mesh.basis(0));
    auto quadrature = make_array<Dim>(subcell_mesh.quadrature(0));
    auto extents = make_array<Dim>(subcell_mesh.extents(0));
    gsl::at(extents, d) = subcell_mesh.extents(0) + 1;
    gsl::at(quadrature, d) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
    const auto face_centered_logical_coords =
        logical_coordinates(face_centered_mesh);
    const auto face_centered_inertial_coords =
        map(face_centered_logical_coords);

    gsl::at(face_centered_gr_vars, d)
        .initialize(face_centered_mesh.number_of_grid_points());
    gsl::at(face_centered_gr_vars, d)
        .assign_subset(soln.variables(
            face_centered_inertial_coords, time,
            typename System::flux_spacetime_variables_tag::tags_list{}));
  }
  return face_centered_gr_vars;
}

std::array<double, 5> test(const size_t num_dg_pts,
                           const ::fd::DerivativeOrder fd_derivative_order,
                           std::optional<double> expansion_velocity) {
  using flux_tags =
      db::wrap_tags_in<::Tags::Flux, typename System::flux_variables,
                       tmpl::size_t<3>, Frame::Inertial>;
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const Affine affine_map{-1.0, 1.0, 7.0, 10.0};
  const auto coordinate_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{affine_map, affine_map, affine_map});
  const auto element = domain::Initialization::create_initial_element(
      ElementId<3>{0, {SegmentId{3, 4}, SegmentId{3, 4}, SegmentId{3, 4}}},
      Block<3>{domain::make_coordinate_map_base<Frame::BlockLogical,
                                                Frame::Inertial>(
                   Affine3D{affine_map, affine_map, affine_map}),
               0,
               {}},
      std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{3, 3, 3}}});

  const grmhd::ValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim
      recons{
          3.8, std::nullopt, 4.0,
          ::fd::reconstruction::FallbackReconstructorType::MonotonisedCentral};
  REQUIRE((static_cast<int>(fd_derivative_order) < 0 or
           (static_cast<size_t>(fd_derivative_order) / 2 <=
            recons.ghost_zone_size())));

  const grmhd::Solutions::BondiMichel soln{1.0, 5.0, 0.05, 1.4, 2.0};

  const double time = 0.0;
  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t num_dg_pts_3d = num_dg_pts * num_dg_pts * num_dg_pts;
  const auto cell_centered_coords =
      coordinate_map(logical_coordinates(subcell_mesh));
  const auto dg_coords = coordinate_map(logical_coordinates(dg_mesh));

  Variables<typename System::spacetime_variables_tag::tags_list>
      cell_centered_spacetime_vars{subcell_mesh.number_of_grid_points()};
  cell_centered_spacetime_vars.assign_subset(
      soln.variables(cell_centered_coords, time,
                     typename System::spacetime_variables_tag::tags_list{}));
  Variables<typename System::primitive_variables_tag::tags_list>
      cell_centered_prim_vars{subcell_mesh.number_of_grid_points()};
  cell_centered_prim_vars.assign_subset(
      soln.variables(cell_centered_coords, time,
                     typename System::primitive_variables_tag::tags_list{}));
  using variables_tag = typename System::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  // Moving mesh info (if needed)
  std::optional<tnsr::I<DataVector, 3>> dg_mesh_velocity{};
  std::optional<tnsr::I<DataVector, 3>> subcell_mesh_velocity{};
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

  // Neighbor data for reconstruction.
  //
  // 0. neighbors coords (our logical coords +2)
  // 1. compute prims from solution
  // 2. compute prims needed for reconstruction
  // 3. set neighbor data
  evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      neighbor_data{};
  using prims_to_reconstruct_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;
  for (const Direction<3>& direction : Direction<3>::all_directions()) {
    auto neighbor_logical_coords = logical_coordinates(subcell_mesh);
    neighbor_logical_coords.get(direction.dimension()) +=
        2.0 * direction.sign();
    auto neighbor_coords = coordinate_map(neighbor_logical_coords);
    const auto neighbor_prims =
        soln.variables(neighbor_coords, time,
                       typename System::primitive_variables_tag::tags_list{});
    static constexpr size_t prim_components =
        Variables<prims_to_reconstruct_tags>::number_of_independent_components;
    DataVector volume_neighbor_data{
        (prim_components +
         ((fd_derivative_order != ::fd::DerivativeOrder::Two)
              ? Variables<flux_tags>::number_of_independent_components
              : 0)) *
            subcell_mesh.number_of_grid_points(),
        0.0};
    if (fd_derivative_order != ::fd::DerivativeOrder::Two) {
      Variables<typename System::spacetime_variables_tag::tags_list>
          neighbor_cell_centered_spacetime_vars{
              subcell_mesh.number_of_grid_points()};
      neighbor_cell_centered_spacetime_vars.assign_subset(soln.variables(
          neighbor_coords, time,
          typename System::spacetime_variables_tag::tags_list{}));
      Variables<typename System::variables_tag::tags_list> neighbor_cons{
          subcell_mesh.number_of_grid_points()};
      apply(make_not_null(&neighbor_cons),
            grmhd::ValenciaDivClean::ConservativeFromPrimitive{},
            neighbor_cell_centered_spacetime_vars, neighbor_prims);

      Variables<flux_tags> neighbor_fluxes{
          std::next(
              volume_neighbor_data.data(),
              static_cast<std::ptrdiff_t>(
                  prim_components * subcell_mesh.number_of_grid_points())),
          Variables<flux_tags>::number_of_independent_components *
              subcell_mesh.number_of_grid_points()};
      apply(make_not_null(&neighbor_fluxes),
            grmhd::ValenciaDivClean::ComputeFluxes{},
            neighbor_cell_centered_spacetime_vars, neighbor_prims,
            neighbor_cons);
      if (expansion_velocity.has_value()) {
        using evolved_vars_tags = typename System::variables_tag::tags_list;
        tnsr::I<DataVector, 3> neighbor_mesh_velocity{
            subcell_mesh.number_of_grid_points()};
        for (size_t i = 0; i < 3; i++) {
          neighbor_mesh_velocity.get(i) =
              neighbor_coords.get(i) * expansion_velocity.value();
        }
        tmpl::for_each<evolved_vars_tags>([&neighbor_cons, &neighbor_fluxes,
                                           &neighbor_mesh_velocity](
                                              auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          using flux_tag = ::Tags::Flux<tag, tmpl::size_t<3>, Frame::Inertial>;
          using FluxTensor = typename flux_tag::type;
          const auto& var = get<tag>(neighbor_cons);
          auto& flux = get<flux_tag>(neighbor_fluxes);
          for (size_t storage_index = 0; storage_index < var.size();
               ++storage_index) {
            const auto tensor_index = var.get_tensor_index(storage_index);
            for (size_t i = 0; i < 3; i++) {
              const auto flux_storage_index =
                  FluxTensor::get_storage_index(prepend(tensor_index, i));
              flux[flux_storage_index] -=
                  var[storage_index] * neighbor_mesh_velocity.get(i);
            }
          }
        });
      }
    }
    Variables<prims_to_reconstruct_tags> prims_to_reconstruct{
        volume_neighbor_data.data(),
        prim_components * subcell_mesh.number_of_grid_points()};
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
            volume_neighbor_data, subcell_mesh.extents(),
            recons.ghost_zone_size(), std::unordered_set{direction.opposite()},
            0)
            .at(direction.opposite());
    const auto key =
        std::pair{direction, *element.neighbors().at(direction).begin()};
    neighbor_data[key] = evolution::dg::subcell::GhostData{1};
    neighbor_data[key].neighbor_ghost_data_for_reconstruction() =
        neighbor_data_in_direction;
  }

  // Below are also dummy variables required for compilation due to boundary
  // condition FD ghost data. Since the element used here for testing has
  // neighbors in all directions, BoundaryConditionGhostData::apply() is not
  // actually called so it is okay to leave these variables somewhat poorly
  // initialized.
  Domain<3> dummy_domain{};
  typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type
      dummy_normal_covector_and_magnitude{};
  const double dummy_time{0.0};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      dummy_functions_of_time{};

  using CellCenteredFluxesTag = evolution::dg::subcell::Tags::CellCenteredFlux<
      typename System::flux_variables, 3>;
  typename CellCenteredFluxesTag::type cell_centered_fluxes{};
  Variables<typename System::variables_tag::tags_list> cell_centered_cons_vars{
      subcell_mesh.number_of_grid_points()};
  apply(make_not_null(&cell_centered_cons_vars),
        grmhd::ValenciaDivClean::ConservativeFromPrimitive{},
        cell_centered_spacetime_vars, cell_centered_prim_vars);
  if (fd_derivative_order != ::fd::DerivativeOrder::Two) {

    cell_centered_fluxes =
        Variables<flux_tags>{subcell_mesh.number_of_grid_points()};
    apply(make_not_null(&(cell_centered_fluxes.value())),
          grmhd::ValenciaDivClean::ComputeFluxes{},
          cell_centered_spacetime_vars, cell_centered_prim_vars,
          cell_centered_cons_vars);
    if (expansion_velocity.has_value()) {
      using evolved_vars_tags = typename System::variables_tag::tags_list;
      tmpl::for_each<evolved_vars_tags>([&cell_centered_cons_vars,
                                         &cell_centered_fluxes,
                                         &subcell_mesh_velocity](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        using flux_tag = ::Tags::Flux<tag, tmpl::size_t<3>, Frame::Inertial>;
        using FluxTensor = typename flux_tag::type;
        const auto& var = get<tag>(cell_centered_cons_vars);
        auto& flux = get<flux_tag>(cell_centered_fluxes.value());
        for (size_t storage_index = 0; storage_index < var.size();
             ++storage_index) {
          const auto tensor_index = var.get_tensor_index(storage_index);
          for (size_t i = 0; i < 3; i++) {
            const auto flux_storage_index =
                FluxTensor::get_storage_index(prepend(tensor_index, i));
            flux[flux_storage_index] -=
                var[storage_index] * subcell_mesh_velocity.value().get(i);
          }
        }
      });
    }
  }
  auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Element<3>, evolution::dg::subcell::Tags::Mesh<3>,
          domain::Tags::Mesh<3>, fd::Tags::Reconstructor,
          evolution::Tags::BoundaryCorrection<grmhd::ValenciaDivClean::System>,
          hydro::Tags::EquationOfState<EquationsOfState::PolytropicFluid<true>>,
          typename System::spacetime_variables_tag,
          typename System::primitive_variables_tag, dt_variables_tag,
          variables_tag,
          evolution::dg::subcell::Tags::OnSubcellFaces<
              typename System::flux_spacetime_variables_tag, 3>,
          evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
          Tags::ConstraintDampingParameter, evolution::dg::Tags::MortarData<3>,
          domain::Tags::ElementMap<3, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::Domain<3>,
          domain::Tags::MeshVelocity<3, Frame::Inertial>,
          domain::Tags::DivMeshVelocity,
          evolution::dg::Tags::NormalCovectorAndMagnitude<3>, ::Tags::Time,
          domain::Tags::FunctionsOfTimeInitialize, DummyAnalyticSolutionTag,
          Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars>,
          CellCenteredFluxesTag,
          evolution::dg::subcell::Tags::SubcellOptions<3>,
          evolution::dg::subcell::Tags::ReconstructionOrder<3>>,
      db::AddComputeTags<
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>>>(
      element, subcell_mesh, dg_mesh,
      std::unique_ptr<grmhd::ValenciaDivClean::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      std::unique_ptr<
          grmhd::ValenciaDivClean::BoundaryCorrections::BoundaryCorrection>{
          std::make_unique<
              grmhd::ValenciaDivClean::BoundaryCorrections::Hll>()},
      soln.equation_of_state(), cell_centered_spacetime_vars,
      cell_centered_prim_vars,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      cell_centered_cons_vars,
      face_centered_gr_tags(subcell_mesh, time, coordinate_map, soln),
      neighbor_data, 1.0, evolution::dg::Tags::MortarData<3>::type{},
      ElementMap<3, Frame::Grid>{
          ElementId<3>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<3>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}),
      std::move(dummy_domain), dg_mesh_velocity, div_dg_mesh_velocity,
      dummy_normal_covector_and_magnitude, dummy_time,
      clone_unique_ptrs(dummy_functions_of_time),
      grmhd::Solutions::SmoothFlow{}, DummyEvolutionMetaVars{},
      cell_centered_fluxes,
      evolution::dg::subcell::SubcellOptions{
          1.0e-7, 1.0e-3, 1.0e-7, 1.0e-3, 4.0, 4.0, false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
          std::nullopt, fd_derivative_order},
      typename evolution::dg::subcell::Tags::ReconstructionOrder<3>::type{});

  db::mutate_apply<ConservativeFromPrimitive>(make_not_null(&box));

  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>
      cell_centered_logical_to_grid_inv_jacobian{};
  const auto cell_centered_logical_to_inertial_inv_jacobian =
      coordinate_map.inv_jacobian(logical_coordinates(subcell_mesh));
  for (size_t i = 0; i < cell_centered_logical_to_grid_inv_jacobian.size();
       ++i) {
    cell_centered_logical_to_grid_inv_jacobian[i] =
        cell_centered_logical_to_inertial_inv_jacobian[i];
  }
  subcell::TimeDerivative::apply(
      make_not_null(&box), cell_centered_logical_to_grid_inv_jacobian,
      determinant(cell_centered_logical_to_grid_inv_jacobian));

  if (static_cast<int>(fd_derivative_order) < 0) {
    CHECK(db::get<evolution::dg::subcell::Tags::ReconstructionOrder<3>>(box)
              .has_value());
  } else {
    CHECK(not db::get<evolution::dg::subcell::Tags::ReconstructionOrder<3>>(box)
                  .has_value());
  }

  // We test that the time derivative converges to zero,
  // so we remove the expected value of the time derivative for moving meshes
  Variables<typename System::variables_tag::tags_list>
      output_minus_expected_dt_cons_vars{subcell_mesh.number_of_grid_points()};
  const auto& dt_vars = db::get<dt_variables_tag>(box);

  tmpl::for_each<typename System::variables_tag::tags_list>(
      [&box, &subcell_mesh, &expansion_velocity, &cell_centered_coords,
       &cell_centered_logical_to_inertial_inv_jacobian,
       &output_minus_expected_dt_cons_vars, &dt_vars](auto cons_var_tag_v) {
        using cons_var_tag = tmpl::type_from<decltype(cons_var_tag_v)>;
        const auto& cons_var = get<cons_var_tag>(box);
        const auto deriv_cons_var =
            partial_derivative(cons_var, subcell_mesh,
                               cell_centered_logical_to_inertial_inv_jacobian);

        auto& output_minus_expected_dt_var =
            get<cons_var_tag>(output_minus_expected_dt_cons_vars);
        const auto& output_dt_var = get<::Tags::dt<cons_var_tag>>(dt_vars);
        for (size_t i = 0; i < output_minus_expected_dt_var.size(); ++i) {
          output_minus_expected_dt_var[i] = output_dt_var[i];
          if (expansion_velocity.has_value()) {
            for (size_t j = 0; j < 3; ++j) {
              const auto deriv_index =
                  j * output_minus_expected_dt_var.size() + i;
              output_minus_expected_dt_var[i] -= cell_centered_coords.get(j) *
                                                 deriv_cons_var[deriv_index] *
                                                 expansion_velocity.value();
            }
          }
        }
      });

  return {
      {max(abs(get(get<Tags::TildeD>(output_minus_expected_dt_cons_vars)))),
       max(abs(get(get<Tags::TildeYe>(output_minus_expected_dt_cons_vars)))),
       max(abs(get(get<Tags::TildeTau>(output_minus_expected_dt_cons_vars)))),
       max(get(
           magnitude(get<Tags::TildeS<>>(output_minus_expected_dt_cons_vars)))),
       max(get(magnitude(
           get<Tags::TildeB<>>(output_minus_expected_dt_cons_vars))))}};
}

// [[TimeOut, 10]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.TimeDerivative",
    "[Unit][Evolution]") {
  std::optional<std::array<double, 5>> previous_error_5{};
  std::optional<std::array<double, 5>> previous_error_6{};
  std::array<double, 5> second_order_error_5{};
  std::array<double, 5> second_order_error_6{};
  std::optional<double> dummy_expansion_velocity{};
  using DO = ::fd::DerivativeOrder;
  for (const DO fd_do : {DO::Two, DO::Four, DO::Six, DO::Eight, DO::Ten}) {
    CAPTURE(fd_do);
    // This tests sets up a cube [2,3]^3 in a Bondi-Michel spacetime and
    // verifies that the time derivative vanishes. Or, more specifically, that
    // the time derivative decreases with increasing resolution.
    const auto five_pts_data = test(5, fd_do, dummy_expansion_velocity);
    const auto six_pts_data = test(6, fd_do, dummy_expansion_velocity);
    if (fd_do == DO::Two) {
      second_order_error_5 = five_pts_data;
      second_order_error_6 = six_pts_data;
    }

    for (size_t i = 0; i < five_pts_data.size(); ++i) {
      CAPTURE(i);
      CHECK(gsl::at(six_pts_data, i) < gsl::at(five_pts_data, i));
      // Check that as we increase the order of the FD derivative the error
      // decreases.
      if (previous_error_5.has_value()) {
        CHECK(gsl::at(five_pts_data, i) < gsl::at(previous_error_5.value(), i));
      }
      if (previous_error_6.has_value()) {
        CHECK(gsl::at(six_pts_data, i) < gsl::at(previous_error_6.value(), i));
      }
    }
    previous_error_5 = five_pts_data;
    previous_error_6 = six_pts_data;
  }

  // Check that a vanishing mesh velocity does not modify the answer
  {
    // This tests sets up a cube [2,3]^3 in a Bondi-Michel spacetime and
    // verifies that the time derivative is the same when using no mesh
    // velocity and when using a zero mesh velocity
    std::optional<double> zero_expansion_velocity(0.0);
    const auto data_no_mesh_velocity =
        test(5, DO::Four, dummy_expansion_velocity);
    const auto data_mesh_velocity = test(5, DO::Four, zero_expansion_velocity);

    for (size_t i = 0; i < data_no_mesh_velocity.size(); ++i) {
      CAPTURE(i);
      CHECK(gsl::at(data_no_mesh_velocity, i) ==
            gsl::at(data_mesh_velocity, i));
    }
  }

  // Now use an expansion map.
  previous_error_5 = {};
  previous_error_6 = {};
  for (const DO fd_do : {DO::Two, DO::Four}) {
    CAPTURE(fd_do);
    std::optional<double> expansion_velocity(0.1);
    // This tests sets up a cube [2,3]^3 in a Bondi-Michel spacetime and
    // verifies that the difference between correct and expected time
    // derivative vanishes. Or, more specifically, that
    // the time derivative decreases with increasing resolution.
    const auto five_pts_data = test(5, fd_do, expansion_velocity);
    const auto six_pts_data = test(6, fd_do, expansion_velocity);
    if (fd_do == DO::Two) {
      second_order_error_5 = five_pts_data;
      second_order_error_6 = six_pts_data;
    }

    for (size_t i = 0; i < five_pts_data.size(); ++i) {
      CAPTURE(i);
      CHECK(gsl::at(six_pts_data, i) < gsl::at(five_pts_data, i));
      // Check that as we increase the order of the FD derivative the error
      // decreases.
      if (previous_error_5.has_value()) {
        CHECK(gsl::at(five_pts_data, i) < gsl::at(previous_error_5.value(), i));
      }
      if (previous_error_6.has_value()) {
        CHECK(gsl::at(six_pts_data, i) < gsl::at(previous_error_6.value(), i));
      }
    }
    previous_error_5 = five_pts_data;
    previous_error_6 = six_pts_data;
  }

  // Check the adaptive correction order works.
  for (const auto& recon_order : make_array(
           DO::OneHigherThanRecons, DO::OneHigherThanReconsButFiveToFour)) {
    CAPTURE(recon_order);
    const auto five_pts_data = test(5, recon_order, dummy_expansion_velocity);
    const auto six_pts_data = test(6, recon_order, dummy_expansion_velocity);
    for (size_t i = 0; i < five_pts_data.size(); ++i) {
      CAPTURE(i);
      CHECK(gsl::at(six_pts_data, i) < gsl::at(five_pts_data, i));
      CHECK(gsl::at(five_pts_data, i) < gsl::at(second_order_error_5, i));
      CHECK(gsl::at(six_pts_data, i) < gsl::at(second_order_error_6, i));
    }
  }
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean
