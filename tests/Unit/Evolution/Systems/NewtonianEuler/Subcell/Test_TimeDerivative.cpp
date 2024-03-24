// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/LaneEmdenGravitationalField.hpp"
#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/AoWeno.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/LaneEmdenStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/PrettyType.hpp"

namespace NewtonianEuler {
namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t Dim>
auto make_coord_map();

template <>
auto make_coord_map<1>() {
  Affine affine_map{-1.0, 1.0, 1.0, 3.0};
  return domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
      affine_map);
}

template <>
auto make_coord_map<2>() {
  Affine affine_map{-1.0, 1.0, 1.0, 3.0};
  Affine2D product_map{affine_map, affine_map};
  return domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
      product_map);
}

template <>
auto make_coord_map<3>() {
  Affine affine_map{-1.0, 1.0, 1.0, 3.0};
  Affine3D product_map{affine_map, affine_map, affine_map};
  return domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
      product_map);
}

template <size_t Dim>
auto make_element();

template <>
auto make_element<1>() {
  Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  return domain::Initialization::create_initial_element(
      ElementId<1>{0, {SegmentId{2, 2}}},
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
      ElementId<2>{0, {SegmentId{2, 2}, SegmentId{2, 2}}},
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
      ElementId<3>{0, {SegmentId{2, 2}, SegmentId{2, 2}, SegmentId{2, 2}}},
      Block<3>{domain::make_coordinate_map_base<Frame::BlockLogical,
                                                Frame::Inertial>(
                   Affine3D{affine_map, affine_map, affine_map}),
               0,
               {}},
      std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{3, 3, 3}}});
}

template <size_t Dim>
struct SmoothFlowMetaVars {
  static constexpr size_t volume_dim = Dim;
  using initial_data = NewtonianEuler::Solutions::SmoothFlow<Dim>;
  using system = NewtonianEuler::System<Dim>;
  using source_term_tag = NewtonianEuler::Tags::SourceTerm<Dim>;
  static auto solution() {
    return initial_data{make_array<Dim>(0.0), make_array<Dim>(-0.2), 0.5, 1.5,
                        0.01};
  }
};

struct LaneEmdenStarMetaVars {
  static constexpr size_t volume_dim = 3;
  using initial_data = NewtonianEuler::Solutions::LaneEmdenStar;
  using system = NewtonianEuler::System<3>;
  using source_term_tag = NewtonianEuler::Tags::SourceTerm<3>;
  static auto solution() { return initial_data{0.7, 250.0}; }
};

template <typename Metavariables>
std::array<double, 3> test(const size_t num_dg_pts) {
  using metavariables = Metavariables;
  static constexpr size_t dim = metavariables::volume_dim;
  using solution = typename metavariables::initial_data;
  using system = typename metavariables::system;
  const auto element = make_element<dim>();
  const ElementMap<dim, Frame::Grid> element_map{
      element.id(), make_coord_map<dim>().get_clone()};
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<dim>{});

  const solution soln = metavariables::solution();

  const double time = 0.0;
  const Mesh<dim> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto cell_centered_coords =
      (*grid_to_inertial_map)(element_map(logical_coordinates(subcell_mesh)));

  using prim_tags = typename system::primitive_variables_tag::tags_list;
  Variables<prim_tags> cell_centered_prim_vars{
      subcell_mesh.number_of_grid_points()};
  cell_centered_prim_vars.assign_subset(
      soln.variables(cell_centered_coords, time, prim_tags{}));
  using variables_tag = typename system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  // Neighbor data for reconstruction.
  //
  // 0. neighbors coords (our logical coords +2)
  // 1. compute prims from solution
  // 2. compute prims needed for reconstruction
  // 3. set neighbor data
  typename evolution::dg::subcell::Tags::GhostDataForReconstruction<dim>::type
      neighbor_data{};
  using prims_to_reconstruct_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, dim>,
                 hydro::Tags::Pressure<DataVector>>;
  for (const Direction<dim>& direction : Direction<dim>::all_directions()) {
    auto neighbor_logical_coords = logical_coordinates(subcell_mesh);
    neighbor_logical_coords.get(direction.dimension()) +=
        2.0 * direction.sign();
    auto neighbor_coords =
        (*grid_to_inertial_map)(element_map(neighbor_logical_coords));
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
            NewtonianEuler::fd::MonotonisedCentralPrim<dim>{}.ghost_zone_size(),
            std::unordered_set{direction.opposite()}, 0, {})
            .at(direction.opposite());
    const auto key = DirectionalId<dim>{
        direction, *element.neighbors().at(direction).begin()};
    neighbor_data[key] = evolution::dg::subcell::GhostData{1};
    neighbor_data[key].neighbor_ghost_data_for_reconstruction() =
        neighbor_data_in_direction;
  }

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      dummy_functions_of_time{};

  auto box = db::create<
      db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<metavariables>,
          typename metavariables::source_term_tag,
          ::Tags::AnalyticSolution<solution>, domain::Tags::Element<dim>,
          domain::Tags::ElementMap<dim, Frame::Grid>,
          evolution::dg::subcell::Tags::Mesh<dim>, fd::Tags::Reconstructor<dim>,
          evolution::Tags::BoundaryCorrection<system>,
          hydro::Tags::EquationOfState<false, 2>,
          typename system::primitive_variables_tag, dt_variables_tag,
          variables_tag,
          evolution::dg::subcell::Tags::GhostDataForReconstruction<dim>,
          evolution::dg::Tags::MortarData<dim>,
          domain::CoordinateMaps::Tags::CoordinateMap<dim, Frame::Grid,
                                                      Frame::Inertial>,
          ::Tags::Time, domain::Tags::FunctionsOfTimeInitialize>,
      db::AddComputeTags<
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<dim>,
          ::domain::Tags::MappedCoordinates<
              ::domain::Tags::ElementMap<dim, Frame::Grid>,
              evolution::dg::subcell::Tags::Coordinates<dim,
                                                        Frame::ElementLogical>,
              evolution::dg::subcell::Tags::Coordinates>,
          evolution::dg::subcell::Tags::InertialCoordinatesCompute<
              ::domain::CoordinateMaps::Tags::CoordinateMap<dim, Frame::Grid,
                                                            Frame::Inertial>>,
          evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGridCompute<
              ::domain::Tags::ElementMap<dim, Frame::Grid>, dim>,
          evolution::dg::subcell::fd::Tags::
              DetInverseJacobianLogicalToGridCompute<dim>,
          evolution::dg::subcell::fd::Tags::
              InverseJacobianLogicalToInertialCompute<
                  ::domain::CoordinateMaps::Tags::CoordinateMap<
                      dim, Frame::Grid, Frame::Inertial>,
                  dim>,
          evolution::dg::subcell::fd::Tags::
              DetInverseJacobianLogicalToInertialCompute<
                  ::domain::CoordinateMaps::Tags::CoordinateMap<
                      dim, Frame::Grid, Frame::Inertial>,
                  dim>>>(
      metavariables{},

      []() -> std::unique_ptr<::NewtonianEuler::Sources::Source<dim>> {
        if constexpr (std::is_same_v<
                          solution,
                          NewtonianEuler::Solutions::SmoothFlow<dim>>) {
          return std::make_unique<::NewtonianEuler::Sources::NoSource<dim>>();
        } else {
          return std::make_unique<
              ::NewtonianEuler::Sources::LaneEmdenGravitationalField>(0.7,
                                                                      250.0);
        }
      }(),

      soln, element,
      ElementMap<dim, Frame::Grid>{element_map.element_id(),
                                   element_map.block_map().get_clone()},
      subcell_mesh,
      std::unique_ptr<NewtonianEuler::fd::Reconstructor<dim>>{
          std::make_unique<NewtonianEuler::fd::MonotonisedCentralPrim<dim>>()},
      std::unique_ptr<
          NewtonianEuler::BoundaryCorrections::BoundaryCorrection<dim>>{
          std::make_unique<NewtonianEuler::BoundaryCorrections::Hll<dim>>()},
      soln.equation_of_state().promote_to_2d_eos(), cell_centered_prim_vars,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      typename variables_tag::type{}, neighbor_data,
      typename evolution::dg::Tags::MortarData<dim>::type{},
      grid_to_inertial_map->get_clone(), time,
      clone_unique_ptrs(dummy_functions_of_time));
  db::mutate_apply<ConservativeFromPrimitive<dim>>(make_not_null(&box));

  NewtonianEuler::subcell::TimeDerivative<dim>::apply(make_not_null(&box));

  const auto& dt_vars = db::get<dt_variables_tag>(box);
  return {{max(abs(get(get<::Tags::dt<Tags::MassDensityCons>>(dt_vars)))),
           max(abs(get(get<::Tags::dt<Tags::EnergyDensity>>(dt_vars)))),
           max(get(magnitude(
               get<::Tags::dt<Tags::MomentumDensity<dim>>>(dt_vars))))}};
}

template <typename Metavariables>
void test_convergence() {
  CAPTURE(pretty_type::get_name<Metavariables>());
  // This tests sets up a stationary SmoothFlow problem or LaneEmdenStar and
  // verifies that the time derivative vanishes. Or, more specifically, that the
  // time derivative decreases with increasing resolution.
  // For the SmoothFlow problem, the time derivatives of the momentum and energy
  // densities are roundoff.
  const auto four_pts_data = test<Metavariables>(4);
  const auto eight_pts_data = test<Metavariables>(8);

  for (size_t i = 0; i < four_pts_data.size(); ++i) {
    CAPTURE(i);
    CAPTURE(gsl::at(four_pts_data, i));
    CAPTURE(gsl::at(eight_pts_data, i));
    const bool converging =
        gsl::at(eight_pts_data, i) < gsl::at(four_pts_data, i);
    const bool roundoff = gsl::at(eight_pts_data, i) < 1.e-14 and
                          gsl::at(four_pts_data, i) < 1.e-14;
    CHECK((converging or roundoff));
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Subcell.TimeDerivative",
    "[Unit][Evolution]") {
  test_convergence<SmoothFlowMetaVars<1>>();
  test_convergence<SmoothFlowMetaVars<2>>();
  test_convergence<SmoothFlowMetaVars<3>>();
  test_convergence<LaneEmdenStarMetaVars>();
}
}  // namespace
}  // namespace NewtonianEuler
