// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SliceVariables.hpp"
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
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/MonotisedCentral.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/NeighborPackagedData.hpp"
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

double test(const size_t num_dg_pts) {
  using variables_tag = typename System::variables_tag;

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  const auto coordinate_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{affine_map, affine_map, affine_map});
  const auto moving_mesh_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});
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
  const auto dg_coords = coordinate_map(logical_coordinates(dg_mesh));

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

  Variables<typename System::spacetime_variables_tag::tags_list>
      dg_spacetime_vars{dg_mesh.number_of_grid_points()};
  dg_spacetime_vars.assign_subset(soln.variables(
      dg_coords, time, typename System::spacetime_variables_tag::tags_list{}));
  Variables<typename System::primitive_variables_tag::tags_list> dg_prim_vars{
      dg_mesh.number_of_grid_points()};
  dg_prim_vars.assign_subset(soln.variables(
      dg_coords, time, typename System::primitive_variables_tag::tags_list{}));

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
    for (size_t i = 0; i < 3; ++i) {
      unnormalized_covector.get(i) =
          coordinate_map.inv_jacobian(face_logical_coords)
              .get(direction.dimension(), i);
    }
    unnormalized_normal_covectors[direction] = unnormalized_covector;
    Variables<tmpl::list<
        inverse_spatial_metric_tag,
        evolution::dg::Actions::detail::NormalVector<3>,
        evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>>
        fields_on_face{face_mesh.number_of_grid_points()};
    fields_on_face.assign_subset(
        soln.variables(coordinate_map(face_logical_coords), time,
                       tmpl::list<inverse_spatial_metric_tag>{}));
    normal_vectors[direction] = std::nullopt;
    evolution::dg::Actions::detail::
        unit_normal_vector_and_covector_and_magnitude<System>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, moving_mesh_map);
  }

  auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Element<3>, domain::Tags::Mesh<3>,
          evolution::dg::subcell::Tags::Mesh<3>, fd::Tags::Reconstructor,
          evolution::Tags::BoundaryCorrection<grmhd::ValenciaDivClean::System>,
          hydro::Tags::EquationOfState<EquationsOfState::PolytropicFluid<true>>,
          typename System::spacetime_variables_tag,
          typename System::primitive_variables_tag, variables_tag,
          evolution::dg::subcell::Tags::OnSubcellFaces<
              typename System::flux_spacetime_variables_tag, 3>,
          evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<
              3>,
          Tags::ConstraintDampingParameter, evolution::dg::Tags::MortarData<3>,
          domain::Tags::MeshVelocity<3>,
          evolution::dg::Tags::NormalCovectorAndMagnitude<3>>,
      db::AddComputeTags<
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>>>(
      element, dg_mesh, subcell_mesh,
      std::unique_ptr<grmhd::ValenciaDivClean::fd::Reconstructor>{
          std::make_unique<
              grmhd::ValenciaDivClean::fd::MonotisedCentralPrim>()},
      std::unique_ptr<
          grmhd::ValenciaDivClean::BoundaryCorrections::BoundaryCorrection>{
          std::make_unique<
              grmhd::ValenciaDivClean::BoundaryCorrections::Hll>()},
      soln.equation_of_state(), dg_spacetime_vars, dg_prim_vars,
      typename variables_tag::type{dg_mesh.number_of_grid_points()},
      face_centered_gr_tags(subcell_mesh, time, coordinate_map, soln),
      neighbor_data, 1.0, evolution::dg::Tags::MortarData<3>::type{},
      std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>{}, normal_vectors);
  db::mutate_apply<ConservativeFromPrimitive>(make_not_null(&box));

  std::vector<std::pair<Direction<3>, ElementId<3>>>
      mortars_to_reconstruct_to{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    mortars_to_reconstruct_to.emplace_back(direction, *neighbors.begin());
  }

  const auto all_packaged_data =
      subcell::NeighborPackagedData::apply(box, mortars_to_reconstruct_to);

  // Parse out evolved vars, since those are easiest to check for correctness,
  // then return absolute difference between analytic and reconstructed values.
  FixedHashMap<maximum_number_of_neighbors(3),
               std::pair<Direction<3>, ElementId<3>>,
               typename variables_tag::type,
               boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      evolved_vars_errors{};
  double max_abs_error = 0.0;
  for (const auto& [direction_and_id, data] : all_packaged_data) {
    const auto& direction = direction_and_id.first;
    using dg_package_field_tags = typename grmhd::ValenciaDivClean::
        BoundaryCorrections::Hll::dg_package_field_tags;
    const Mesh<2> face_mesh = dg_mesh.slice_away(direction.dimension());
    Variables<dg_package_field_tags> packaged_data{
        face_mesh.number_of_grid_points()};
    std::copy(data.begin(), data.end(), packaged_data.data());

    auto sliced_vars = data_on_slice(
        db::get<variables_tag>(box), dg_mesh.extents(), direction.dimension(),
        direction.side() == Side::Upper
            ? dg_mesh.extents(direction.dimension()) - 1
            : 0);

    tmpl::for_each<typename variables_tag::type::tags_list>(
        [&sliced_vars, &max_abs_error, &packaged_data](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          auto& sliced_tensor = get<tag>(sliced_vars);
          const auto& packaged_data_tensor = get<tag>(packaged_data);
          for (size_t tensor_index = 0; tensor_index < sliced_tensor.size();
               ++tensor_index) {
            max_abs_error = std::max(
                max_abs_error, max(abs(sliced_tensor[tensor_index] -
                                       packaged_data_tensor[tensor_index])));
          }
        });
  }
  return max_abs_error;
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.NeighborPackagedData",
    "[Unit][Evolution]") {
  // This tests sets up a cube [2,3]^3 in a Bondi-Michel spacetime and verifies
  // that the difference between the reconstructed evolved variables and the
  // sliced (exact on LGL grid) evolved variables on the interfaces decreases.
  CHECK(test(4) > test(8));
  // Check that the error is "reasonably small"
  CHECK(test(4) < 1.0e-3);
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean
