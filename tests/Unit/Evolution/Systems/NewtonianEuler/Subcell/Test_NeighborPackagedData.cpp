// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/AoWeno.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/SmoothFlow.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t Dim>
auto make_coord_map();

template <>
auto make_coord_map<1>() {
  Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      affine_map);
}

template <>
auto make_coord_map<2>() {
  Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  Affine2D product_map{affine_map, affine_map};
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      product_map);
}

template <>
auto make_coord_map<3>() {
  Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  Affine3D product_map{affine_map, affine_map, affine_map};
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      product_map);
}

template <size_t Dim>
auto make_element();

template <>
auto make_element<1>() {
  Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  return domain::Initialization::create_initial_element(
      ElementId<1>{0, {SegmentId{3, 4}}},
      Block<1>{domain::make_coordinate_map_base<Frame::BlockLogical,
                                                Frame::Inertial>(affine_map),
               0,
               {}},
      std::vector<std::array<size_t, 1>>{std::array<size_t, 1>{{3}}});
}

template <>
auto make_element<2>() {
  Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  return domain::Initialization::create_initial_element(
      ElementId<2>{0, {SegmentId{3, 4}, SegmentId{3, 4}}},
      Block<2>{domain::make_coordinate_map_base<Frame::BlockLogical,
                                                Frame::Inertial>(
                   Affine2D{affine_map, affine_map}),
               0,
               {}},
      std::vector<std::array<size_t, 2>>{std::array<size_t, 2>{{3, 3}}});
}

template <>
auto make_element<3>() {
  Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  return domain::Initialization::create_initial_element(
      ElementId<3>{0, {SegmentId{3, 4}, SegmentId{3, 4}, SegmentId{3, 4}}},
      Block<3>{domain::make_coordinate_map_base<Frame::BlockLogical,
                                                Frame::Inertial>(
                   Affine3D{affine_map, affine_map, affine_map}),
               0,
               {}},
      std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{3, 3, 3}}});
}

template <size_t Dim>
struct MetaVars {
  using solution = NewtonianEuler::Solutions::SmoothFlow<Dim>;
  using system = NewtonianEuler::System<Dim>;
};

template <size_t Dim>
double test(const size_t num_dg_pts) {
  using solution = NewtonianEuler::Solutions::SmoothFlow<Dim>;
  using eos = typename solution::equation_of_state_type;
  using system = NewtonianEuler::System<Dim>;
  using variables_tag = typename system::variables_tag;
  const auto coordinate_map = make_coord_map<Dim>();
  const auto moving_mesh_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{});
  const auto element = make_element<Dim>();

  const solution soln{make_array<Dim>(0.3), make_array<Dim>(-0.2), 0.5, 1.5,
                      0.01};

  const double time = 0.0;
  const Mesh<Dim> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto dg_coords = coordinate_map(logical_coordinates(dg_mesh));

  // Neighbor data for reconstruction.
  //
  // 0. neighbors coords (our logical coords +2)
  // 1. compute prims from solution
  // 2. compute prims needed for reconstruction
  // 3. set neighbor data
  typename evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>::type
      neighbor_data{};
  using prims_to_reconstruct_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::Pressure<DataVector>>;
  using prim_tags = typename system::primitive_variables_tag::tags_list;
  for (const Direction<Dim>& direction : Direction<Dim>::all_directions()) {
    auto neighbor_logical_coords = logical_coordinates(subcell_mesh);
    neighbor_logical_coords.get(direction.dimension()) +=
        2.0 * direction.sign();
    auto neighbor_coords = coordinate_map(neighbor_logical_coords);
    const auto neighbor_prims =
        soln.variables(neighbor_coords, time, prim_tags{});
    Variables<prims_to_reconstruct_tags> prims_to_reconstruct{
        subcell_mesh.number_of_grid_points()};
    tmpl::for_each<prims_to_reconstruct_tags>(
        [&prims_to_reconstruct, &neighbor_prims](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(prims_to_reconstruct) = get<tag>(neighbor_prims);
        });
    // Slice data so we can add it to the element's neighbor data
    DataVector neighbor_data_in_direction =
        evolution::dg::subcell::slice_data(
            prims_to_reconstruct, subcell_mesh.extents(),
            NewtonianEuler::fd::MonotonisedCentralPrim<Dim>{}.ghost_zone_size(),
            std::unordered_set{direction.opposite()}, 0, {})
            .at(direction.opposite());
    const auto key = DirectionalId<Dim>{
        direction, *element.neighbors().at(direction).begin()};
    neighbor_data[key] = evolution::dg::subcell::GhostData{1};
    neighbor_data[key].neighbor_ghost_data_for_reconstruction() =
        neighbor_data_in_direction;
  }
  Variables<prim_tags> dg_prim_vars{dg_mesh.number_of_grid_points()};
  dg_prim_vars.assign_subset(soln.variables(dg_coords, time, prim_tags{}));
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normal_vectors{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const Mesh<Dim - 1> face_mesh = dg_mesh.slice_away(direction.dimension());
    const auto face_logical_coords =
        interface_logical_coordinates(face_mesh, direction);
    std::unordered_map<Direction<Dim>,
                       tnsr::i<DataVector, Dim, Frame::Inertial>>
        unnormalized_normal_covectors{};
    tnsr::i<DataVector, Dim, Frame::Inertial> unnormalized_covector{};
    for (size_t i = 0; i < Dim; ++i) {
      unnormalized_covector.get(i) =
          coordinate_map.inv_jacobian(face_logical_coords)
              .get(direction.dimension(), i);
    }
    unnormalized_normal_covectors[direction] = unnormalized_covector;
    Variables<tmpl::list<
        evolution::dg::Actions::detail::NormalVector<Dim>,
        evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>>
        fields_on_face{face_mesh.number_of_grid_points()};
    normal_vectors[direction] = std::nullopt;
    evolution::dg::Actions::detail::
        unit_normal_vector_and_covector_and_magnitude<system>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, moving_mesh_map);
  }
  auto box = db::create<
      db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<MetaVars<Dim>>,
          domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
          evolution::dg::subcell::Tags::Mesh<Dim>,
          NewtonianEuler::fd::Tags::Reconstructor<Dim>,
          evolution::Tags::BoundaryCorrection<system>,
          hydro::Tags::EquationOfState<false, eos::thermodynamic_dim>,
          typename system::primitive_variables_tag, variables_tag,
          evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
          evolution::dg::Tags::MortarData<Dim>, domain::Tags::MeshVelocity<Dim>,
          evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
          evolution::dg::subcell::Tags::SubcellOptions<Dim>>,
      db::AddComputeTags<
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<Dim>>>(
      MetaVars<Dim>{}, element, dg_mesh, subcell_mesh,
      std::unique_ptr<NewtonianEuler::fd::Reconstructor<Dim>>{
          std::make_unique<NewtonianEuler::fd::MonotonisedCentralPrim<Dim>>()},
      std::unique_ptr<
          NewtonianEuler::BoundaryCorrections::BoundaryCorrection<Dim>>{
          std::make_unique<NewtonianEuler::BoundaryCorrections::Hll<Dim>>()},
      soln.equation_of_state().get_clone(), dg_prim_vars,
      typename variables_tag::type{dg_mesh.number_of_grid_points()},
      neighbor_data, typename evolution::dg::Tags::MortarData<Dim>::type{},
      std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>{},
      normal_vectors,
      evolution::dg::subcell::SubcellOptions{
          4.0, 1_st, 1.0e-3, 1.0e-4, false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
          std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1});

  db::mutate_apply<NewtonianEuler::ConservativeFromPrimitive<Dim>>(
      make_not_null(&box));

  std::vector<DirectionalId<Dim>> mortars_to_reconstruct_to{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    mortars_to_reconstruct_to.emplace_back(
        DirectionalId<Dim>{direction, *neighbors.begin()});
  }

  const auto all_packaged_data =
      NewtonianEuler::subcell::NeighborPackagedData::apply(
          box, mortars_to_reconstruct_to);

  // Parse out evolved vars, since those are easiest to check for correctness,
  // then return absolute difference between analytic and reconstructed values.
  DirectionalIdMap<Dim, typename variables_tag::type> evolved_vars_errors{};
  double max_abs_error = 0.0;
  for (const auto& [direction_and_id, data] : all_packaged_data) {
    const auto& direction = direction_and_id.direction();
    using dg_package_field_tags =
        typename NewtonianEuler::BoundaryCorrections::Hll<
            Dim>::dg_package_field_tags;
    const Mesh<Dim - 1> face_mesh = dg_mesh.slice_away(direction.dimension());
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
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtoniainEuler.Subcell.NeighborPackagedData",
    "[Unit][Evolution]") {
  // This tests sets up a cube [2,3]^3 for a SmoothFlow problem and verifies
  // that the difference between the reconstructed evolved variables and the
  // sliced (exact on LGL grid) evolved variables on the interfaces decreases.
  CHECK(test<1>(3) > test<1>(6));
  CHECK(test<2>(3) > test<2>(6));
  CHECK(test<3>(3) > test<3>(6));
  // Check that the error is "reasonably small"
  CHECK(test<1>(4) < 1.0e-4);
  CHECK(test<2>(4) < 1.0e-4);
  CHECK(test<3>(4) < 1.0e-4);
}
