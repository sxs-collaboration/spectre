// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/BoundaryGhostData.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler {
namespace {

// use smoothflow solution for test
template <size_t Dim>
Solutions::SmoothFlow<Dim> make_solution() {
  const double pressure{1.0};
  const double adiabatic_index{1.4};
  const double perturbation_size{0.3};
  std::array<double, Dim> mean_velocity;
  std::array<double, Dim> wave_vector;

  if constexpr (Dim == 1) {
    mean_velocity = {1.0};
    wave_vector = {1.0};
  } else if constexpr (Dim == 2) {
    mean_velocity = {1.0, 1.0};
    wave_vector = {1.0, 1.0};
  } else if constexpr (Dim == 3) {
    mean_velocity = {1.0, 1.0, 1.0};
    wave_vector = {1.0, 1.0, 1.0};
  }
  return Solutions::SmoothFlow<Dim>{mean_velocity, wave_vector, pressure,
                                    adiabatic_index, perturbation_size};
}

template <size_t Dim, typename InitialData>
struct EvolutionMetaVars {
  using system = System<Dim, InitialData>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<BoundaryConditions::BoundaryCondition<Dim>,
                   BoundaryConditions::standard_boundary_conditions<Dim>>>;
  };
};

template <size_t Dim, typename BoundaryConditionType>
void test(const BoundaryConditionType boundary_condition) {
  const size_t num_dg_pts = 5;

  // Create domain and element
  // Let's use [0, 1]^D for test.
  Domain<Dim> domain{};
  std::array<size_t, Dim> refinement_levels{};

  if constexpr (Dim == 1) {
    std::array<double, Dim> lower_bounds{0.0};
    std::array<double, Dim> upper_bounds{1.0};
    std::array<size_t, Dim> number_of_grid_points{num_dg_pts};
    refinement_levels = {0};

    const auto interval = domain::creators::Interval(
        lower_bounds, upper_bounds, refinement_levels, number_of_grid_points,
        std::make_unique<BoundaryConditionType>(boundary_condition),
        std::make_unique<BoundaryConditionType>(boundary_condition), nullptr);
    domain = interval.create_domain();
  } else if constexpr (Dim == 2) {
    std::array<double, Dim> lower_bounds{0.0, 0.0};
    std::array<double, Dim> upper_bounds{1.0, 1.0};
    std::array<size_t, Dim> number_of_grid_points{num_dg_pts, num_dg_pts};
    refinement_levels = {0, 0};

    const auto rectangle = domain::creators::Rectangle(
        lower_bounds, upper_bounds, refinement_levels, number_of_grid_points,
        std::make_unique<BoundaryConditionType>(boundary_condition), nullptr);
    domain = rectangle.create_domain();
  } else if constexpr (Dim == 3) {
    std::array<double, Dim> lower_bounds{0.0, 0.0, 0.0};
    std::array<double, Dim> upper_bounds{1.0, 1.0, 1.0};
    std::array<size_t, Dim> number_of_grid_points{num_dg_pts, num_dg_pts,
                                                  num_dg_pts};
    refinement_levels = {0, 0, 0};

    const auto brick = domain::creators::Brick(
        lower_bounds, upper_bounds, refinement_levels, number_of_grid_points,
        std::make_unique<BoundaryConditionType>(boundary_condition), nullptr);
    domain = brick.create_domain();
  }

  const auto element = domain::Initialization::create_initial_element(
      ElementId<Dim>{0, {SegmentId{0, 0}}}, domain.blocks().at(0),
      std::vector<std::array<size_t, Dim>>{{refinement_levels}});

  // Mesh and coordinates
  const Mesh<Dim> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  // use MC reconstruction for test
  using ReconstructorForTest = fd::MonotisedCentralPrim<Dim>;

  // dummy neighbor data to put into DataBox
  typename evolution::dg::subcell::Tags::NeighborDataForReconstruction<
      Dim>::type neighbor_data{};

  // Below are tags required by DirichletAnalytic boundary condition.
  //  - time
  //  - functions of time
  //  - element map
  //  - coordinate map
  //  - subcell logical coordinates
  //  - analytic solution

  const double time{0.3};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  const ElementMap<Dim, Frame::Grid> logical_to_grid_map(
      ElementId<Dim>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<Dim>{}));

  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{});

  const auto subcell_logical_coords = logical_coordinates(subcell_mesh);

  using SolutionForTest = Solutions::SmoothFlow<Dim>;
  auto solution = make_solution<Dim>();

  // Below are tags required by Outflow boundary condition
  //  - primitive variables on subcell mesh
  //  - equation of state (for computing sound speed)
  //  - volume mesh velocity
  //  - normal vector on face
  using MassDensity = Tags::MassDensity<DataVector>;
  using Velocity = Tags::Velocity<DataVector, Dim>;
  using Pressure = Tags::Pressure<DataVector>;
  using SpecificInternalEnergy = Tags::SpecificInternalEnergy<DataVector>;

  const auto subcell_inertial_coords = (*grid_to_inertial_map)(
      logical_to_grid_map(subcell_logical_coords), time, functions_of_time);

  auto volume_prims = [&solution, &subcell_inertial_coords, &time]() {
    return solution.variables(
        subcell_inertial_coords, time,
        tmpl::list<MassDensity, Velocity, Pressure, SpecificInternalEnergy>{});
  }();
  auto& mass_density = get<MassDensity>(volume_prims);
  auto& velocity = get<Velocity>(volume_prims);
  auto& pressure = get<Pressure>(volume_prims);
  auto& specific_internal_energy = get<SpecificInternalEnergy>(volume_prims);

  using EosType = EquationsOfState::PolytropicFluid<false>;
  EosType eos_polytrope(0.1, 4.0 / 3.0);

  std::optional<tnsr::I<DataVector, Dim>> volume_mesh_velocity{};

  typename evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>::type
      normal_vectors{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<Dim>{});
    const auto moving_mesh_map =
        domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Identity<Dim>{});

    // Note that we used `subcell_mesh` here, since we need normal vector field
    // on the `subcell face mesh` to apply outflow condition.
    const Mesh<Dim - 1> face_mesh =
        subcell_mesh.slice_away(direction.dimension());
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
        unit_normal_vector_and_covector_and_magnitude<
            System<Dim, Solutions::SmoothFlow<Dim>>>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, moving_mesh_map);
  }

  // create a box
  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<
          EvolutionMetaVars<Dim, SolutionForTest>>,
      domain::Tags::Domain<Dim>, evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>,
      evolution::dg::subcell::Tags::NeighborDataForReconstruction<Dim>,
      fd::Tags::Reconstructor<Dim>, domain::Tags::MeshVelocity<Dim>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>, ::Tags::Time,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      hydro::Tags::EquationOfState<EosType>,
      ::Tags::AnalyticSolution<SolutionForTest>, MassDensity, Velocity,
      Pressure, SpecificInternalEnergy>>(
      EvolutionMetaVars<Dim, SolutionForTest>{}, std::move(domain),
      subcell_mesh, subcell_logical_coords, neighbor_data,
      std::unique_ptr<fd::Reconstructor<Dim>>{
          std::make_unique<ReconstructorForTest>()},
      volume_mesh_velocity, normal_vectors, time,
      clone_unique_ptrs(functions_of_time),
      ElementMap<Dim, Frame::Grid>{
          ElementId<Dim>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<Dim>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{}),
      eos_polytrope, solution, mass_density, velocity, pressure,
      specific_internal_energy);

  // compute FD ghost data and retrieve the result
  fd::BoundaryGhostData::apply(make_not_null(&box), element,
                               ReconstructorForTest{});

  const auto direction = Direction<Dim>::upper_xi();
  const size_t dim_direction = direction.dimension();

  const std::pair mortar_id = {direction,
                               ElementId<Dim>::external_boundary_id()};
  const std::vector<double>& fd_ghost_data =
      get<evolution::dg::subcell::Tags::NeighborDataForReconstruction<Dim>>(box)
          .at(mortar_id);

  const size_t ghost_zone_size = ReconstructorForTest{}.ghost_zone_size();
  const size_t num_face_pts =
      subcell_mesh.slice_away(0).number_of_grid_points();
  const size_t num_ghost_pts = num_face_pts * ghost_zone_size;

  // copy the returned std::vector<double> data into a Variables to do checks a
  // bit easier
  Variables<tmpl::list<MassDensity, Velocity, Pressure>> fd_ghost_vars{
      num_ghost_pts};
  std::copy(fd_ghost_data.begin(), fd_ghost_data.end(), fd_ghost_vars.data());

  // Now we compare results
  Index<Dim> ghost_data_extents = subcell_mesh.extents();
  ghost_data_extents[dim_direction] = ghost_zone_size;

  if (typeid(BoundaryConditionType) ==
      typeid(BoundaryConditions::DirichletAnalytic<Dim>)) {
    // Slice subcell logical coordinates and shift it to get inertial
    // coordinates of ghost points
    auto sliced_logical_coords =
        evolution::dg::subcell::slice_tensor_for_subcell(
            subcell_logical_coords, subcell_mesh.extents(), ghost_zone_size,
            direction);
    const double delta_x{get<0>(subcell_logical_coords)[1] -
                         get<0>(subcell_logical_coords)[0]};
    sliced_logical_coords.get(dim_direction) =
        sliced_logical_coords.get(dim_direction) +
        direction.sign() * delta_x * ghost_zone_size;
    const auto shifted_inertial_coords = (*grid_to_inertial_map)(
        logical_to_grid_map(sliced_logical_coords), time, functions_of_time);

    // Compute analytic solution
    auto expected_ghost_prims = [&shifted_inertial_coords, &solution, &time]() {
      return solution.variables(shifted_inertial_coords, time,
                                tmpl::list<MassDensity, Velocity, Pressure>{});
    }();

    // Check results
    tmpl::for_each<tmpl::list<MassDensity, Velocity, Pressure>>(
        [&expected_ghost_prims, &fd_ghost_vars](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          CHECK_ITERABLE_APPROX(get<tag>(expected_ghost_prims),
                                get<tag>(fd_ghost_vars));
        });
  }
  if (typeid(BoundaryConditionType) ==
      typeid(BoundaryConditions::Reflection<Dim>)) {
    // Invert the normal component of velocity.
    velocity.get(dim_direction) =
        velocity.get(dim_direction) - direction.sign() * 2.0;

    tmpl::for_each<tmpl::list<MassDensity, Velocity, Pressure>>(
        [&dim_direction, &fd_ghost_vars, &ghost_data_extents, &ghost_zone_size,
         &subcell_mesh, &volume_prims](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;

          for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
            // i-th slice of ghost data
            auto i_th_slice_ghost =
                data_on_slice(get<tag>(fd_ghost_vars), ghost_data_extents,
                              dim_direction, i_ghost);

            // (n-i)-th slice of volume data where n is the extent of subcell
            // mesh along the `direction`. This is the i-th slice when counted
            // from the external boundary.
            auto n_minus_i_th_slice_volume = data_on_slice(
                get<tag>(volume_prims), subcell_mesh.extents(), dim_direction,
                subcell_mesh.extents(dim_direction) - i_ghost - 1);

            // Check results
            CHECK(i_th_slice_ghost == n_minus_i_th_slice_volume);
          }
        });
  }
  if (typeid(BoundaryConditionType) ==
      typeid(BoundaryConditions::Outflow<Dim>)) {
    tmpl::for_each<tmpl::list<MassDensity, Velocity, Pressure>>(
        [&dim_direction, &fd_ghost_vars, &ghost_data_extents, &ghost_zone_size,
         &subcell_mesh, &volume_prims](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;

          for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
            // i-th slice of ghost data
            auto i_th_slice_ghost =
                data_on_slice(get<tag>(fd_ghost_vars), ghost_data_extents,
                              dim_direction, i_ghost);

            // n-th slice of volume data where n is the extent of subcell
            // mesh along the `direction`. This is the data on the outermost FD
            // cell.
            auto n_th_slice_volume = data_on_slice(
                get<tag>(volume_prims), subcell_mesh.extents(), dim_direction,
                subcell_mesh.extents(dim_direction) - 1);

            // Check results
            CHECK(i_th_slice_ghost == n_th_slice_volume);
          }
        });
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Fd.BoundaryGhostData",
                  "[Unit][Evolution]") {
  test<1>(BoundaryConditions::DirichletAnalytic<1>{});
  test<2>(BoundaryConditions::DirichletAnalytic<2>{});
  test<3>(BoundaryConditions::DirichletAnalytic<3>{});

  test<1>(BoundaryConditions::Reflection<1>{});
  test<2>(BoundaryConditions::Reflection<2>{});
  test<3>(BoundaryConditions::Reflection<3>{});

  test<1>(BoundaryConditions::Outflow<1>{});
  test<2>(BoundaryConditions::Outflow<2>{});
  test<3>(BoundaryConditions::Outflow<3>{});
}
}  // namespace
}  // namespace NewtonianEuler
