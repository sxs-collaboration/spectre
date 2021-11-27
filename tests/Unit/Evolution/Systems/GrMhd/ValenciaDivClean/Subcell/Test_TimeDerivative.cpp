// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/MonotisedCentral.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace grmhd::ValenciaDivClean {
namespace {
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

std::array<double, 4> test(const size_t num_dg_pts) {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const Affine affine_map{-1.0, 1.0, 8.0, 9.0};
  const auto coordinate_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{affine_map, affine_map, affine_map});
  const auto element = domain::Initialization::create_initial_element(
      ElementId<3>{0, {SegmentId{3, 4}, SegmentId{3, 4}, SegmentId{3, 4}}},
      Block<3>{domain::make_coordinate_map_base<Frame::BlockLogical,
                                                Frame::Inertial>(
                   Affine3D{affine_map, affine_map, affine_map}),
               0,
               {},
               {}},
      std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{3, 3, 3}}});

  const grmhd::Solutions::BondiMichel soln{1.0, 5.0, 0.05, 1.4, 2.0};

  const double time = 0.0;
  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto cell_centered_coords =
      coordinate_map(logical_coordinates(subcell_mesh));

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

  // Neighbor data for reconstruction.
  //
  // 0. neighbors coords (our logical coords +2)
  // 1. compute prims from solution
  // 2. compute prims needed for reconstruction
  // 3. set neighbor data
  evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<3>::type
      neighbor_data{};
  using prims_to_reconstruct_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
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
    Variables<prims_to_reconstruct_tags> prims_to_reconstruct{
        subcell_mesh.number_of_grid_points()};
    get<hydro::Tags::RestMassDensity<DataVector>>(prims_to_reconstruct) =
        get<hydro::Tags::RestMassDensity<DataVector>>(neighbor_prims);
    get<hydro::Tags::Pressure<DataVector>>(prims_to_reconstruct) =
        get<hydro::Tags::Pressure<DataVector>>(neighbor_prims);
    get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
        prims_to_reconstruct) =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(neighbor_prims);
    for (auto& component :
         get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
             prims_to_reconstruct)) {
      component *=
          get(get<hydro::Tags::LorentzFactor<DataVector>>(neighbor_prims));
    }
    get<hydro::Tags::MagneticField<DataVector, 3>>(prims_to_reconstruct) =
        get<hydro::Tags::MagneticField<DataVector, 3>>(neighbor_prims);
    get<hydro::Tags::DivergenceCleaningField<DataVector>>(
        prims_to_reconstruct) =
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(neighbor_prims);

    // Slice data so we can add it to the element's neighbor data
    DirectionMap<3, bool> directions_to_slice{};
    directions_to_slice[direction.opposite()] = true;
    evolution::dg::subcell::NeighborData neighbor_data_in_direction{};
    neighbor_data_in_direction.data_for_reconstruction =
        evolution::dg::subcell::slice_data(
            prims_to_reconstruct, subcell_mesh.extents(),
            grmhd::ValenciaDivClean::fd::MonotisedCentralPrim{}
                .ghost_zone_size(),
            directions_to_slice)
            .at(direction.opposite());
    neighbor_data[std::pair{direction,
                            *element.neighbors().at(direction).begin()}] =
        neighbor_data_in_direction;
  }

  auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Element<3>, evolution::dg::subcell::Tags::Mesh<3>,
          fd::Tags::Reconstructor,
          evolution::Tags::BoundaryCorrection<grmhd::ValenciaDivClean::System>,
          hydro::Tags::EquationOfState<EquationsOfState::PolytropicFluid<true>>,
          typename System::spacetime_variables_tag,
          typename System::primitive_variables_tag, dt_variables_tag,
          variables_tag,
          evolution::dg::subcell::Tags::OnSubcellFaces<
              typename System::flux_spacetime_variables_tag, 3>,
          evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<
              3>,
          Tags::ConstraintDampingParameter, evolution::dg::Tags::MortarData<3>>,
      db::AddComputeTags<
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>>>(
      element, subcell_mesh,
      std::unique_ptr<grmhd::ValenciaDivClean::fd::Reconstructor>{
          std::make_unique<
              grmhd::ValenciaDivClean::fd::MonotisedCentralPrim>()},
      std::unique_ptr<
          grmhd::ValenciaDivClean::BoundaryCorrections::BoundaryCorrection>{
          std::make_unique<
              grmhd::ValenciaDivClean::BoundaryCorrections::Hll>()},
      soln.equation_of_state(), cell_centered_spacetime_vars,
      cell_centered_prim_vars,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      typename variables_tag::type{},
      face_centered_gr_tags(subcell_mesh, time, coordinate_map, soln),
      neighbor_data, 1.0, evolution::dg::Tags::MortarData<3>::type{});
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

  const auto& dt_vars = db::get<dt_variables_tag>(box);
  return {{max(abs(get(get<::Tags::dt<Tags::TildeD>>(dt_vars)))),
           max(abs(get(get<::Tags::dt<Tags::TildeTau>>(dt_vars)))),
           max(get(magnitude(get<::Tags::dt<Tags::TildeS<>>>(dt_vars)))),
           max(get(magnitude(get<::Tags::dt<Tags::TildeB<>>>(dt_vars))))}};
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.TimeDerivative",
    "[Unit][Evolution]") {
  // This tests sets up a cube [2,3]^3 in a Bondi-Michel spacetime and verifies
  // that the time derivative vanishes. Or, more specifically, that the time
  // derivative decreases with increasing resolution.
  const auto four_pts_data = test(4);
  const auto eight_pts_data = test(8);

  for (size_t i = 0; i < four_pts_data.size(); ++i) {
    CHECK(gsl::at(eight_pts_data, i) < gsl::at(four_pts_data, i));
  }
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean
