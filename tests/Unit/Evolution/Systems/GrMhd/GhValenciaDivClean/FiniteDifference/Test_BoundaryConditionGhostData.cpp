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

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/ConstraintPreservingFreeOutflow.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/ProductOfConditions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean {
namespace {

// Metavariables to parse the list of derived classes of boundary conditions
struct EvolutionMetaVars {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        BoundaryConditions::BoundaryCondition,
        tmpl::push_back<BoundaryConditions::standard_fd_boundary_conditions,
                        BoundaryConditions::DirichletAnalytic>>>;
  };
};

template <typename BoundaryConditionType>
void test(const BoundaryConditionType& boundary_condition) {
  const size_t num_dg_pts = 4;

  // Create a 3D element [-1, 1]^3 and use it for test
  const std::array lower_bounds{-1.0, -1.0, -1.0};
  const std::array upper_bounds{1.0, 1.0, 1.0};
  const std::array<size_t, 3> refinement_levels{0, 0, 0};
  const std::array<size_t, 3> number_of_grid_points{num_dg_pts, num_dg_pts,
                                                    num_dg_pts};
  const auto brick = domain::creators::Brick(
      lower_bounds, upper_bounds, refinement_levels, number_of_grid_points,
      std::make_unique<BoundaryConditionType>(boundary_condition),
      std::make_unique<BoundaryConditionType>(boundary_condition),
      std::make_unique<BoundaryConditionType>(boundary_condition), nullptr);
  auto domain = brick.create_domain();
  auto boundary_conditions = brick.external_boundary_conditions();
  const auto element = domain::Initialization::create_initial_element(
      ElementId<3>{0, {SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}},
      domain.blocks().at(0),
      std::vector<std::array<size_t, 3>>{{refinement_levels}});

  // Mesh and coordinates
  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  // use MC reconstruction for test
  using ReconstructorForTest = fd::MonotonisedCentralPrim;
  const size_t ghost_zone_size{ReconstructorForTest{}.ghost_zone_size()};

  // dummy neighbor data to put into DataBox
  typename evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      neighbor_data{};

  // Below are tags required by DirichletAnalytic boundary condition to compute
  // inertial coords of ghost FD cells:
  //  - time
  //  - functions of time
  //  - element map
  //  - coordinate map
  //  - subcell logical coordinates
  //  - analytic solution

  const double time{0.5};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  const ElementMap<3, Frame::Grid> logical_to_grid_map(
      ElementId<3>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<3>{}));

  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});

  const auto subcell_logical_coords = logical_coordinates(subcell_mesh);

  const gh::Solutions::WrappedGr solution{RelativisticEuler::Solutions::TovStar{
      1.28e-3, EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.get_clone(),
      RelativisticEuler::Solutions::TovCoordinates::Schwarzschild}};
  using SolutionForTest = std::decay_t<decltype(solution)>;

  // Below are tags used for testing Outflow boundary condition
  //  - volume metric and primitive variables on subcell mesh
  //  - mesh velocity
  //  - normal vectors

  const auto subcell_inertial_coords = (*grid_to_inertial_map)(
      logical_to_grid_map(subcell_logical_coords), time, functions_of_time);

  using SpacetimeMetric = gr::Tags::SpacetimeMetric<DataVector, 3>;
  using Pi = gh::Tags::Pi<DataVector, 3>;
  using Phi = gh::Tags::Phi<DataVector, 3>;
  using RestMassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using ElectronFraction = hydro::Tags::ElectronFraction<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using LorentzFactor = hydro::Tags::LorentzFactor<DataVector>;
  using SpatialVelocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;

  std::optional<tnsr::I<DataVector, 3>> volume_mesh_velocity{};

  typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type
      normal_vectors{};
  for (const auto& direction : Direction<3>::all_directions()) {
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<3>{});
    const auto moving_mesh_map =
        domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Identity<3>{});

    using inverse_spatial_metric_tag =
        typename System::inverse_spatial_metric_tag;
    const Mesh<2> face_mesh = subcell_mesh.slice_away(direction.dimension());
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
        solution.variables(coordinate_map(face_logical_coords), time,
                           tmpl::list<inverse_spatial_metric_tag>{}));
    normal_vectors[direction] = std::nullopt;
    evolution::dg::Actions::detail::
        unit_normal_vector_and_covector_and_magnitude<System>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, moving_mesh_map);
  }

  Variables<typename System::primitive_variables_tag::tags_list>
      volume_prim_vars{subcell_mesh.number_of_grid_points()};

  get(get<RestMassDensity>(volume_prim_vars)) = 1.0;
  get(get<ElectronFraction>(volume_prim_vars)) = 0.1;
  get(get<Pressure>(volume_prim_vars)) = 1.0;
  get(get<LorentzFactor>(volume_prim_vars)) = 2.0;
  for (size_t i = 0; i < 3; ++i) {
    get<SpatialVelocity>(volume_prim_vars).get(i) = 0.1;
    get<MagneticField>(volume_prim_vars).get(i) = 0.5;
  }
  get(get<DivergenceCleaningField>(volume_prim_vars)) = 1e-2;

  // create a box for test
  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<EvolutionMetaVars>,
      domain::Tags::Domain<3>, domain::Tags::ExternalBoundaryConditions<3>,
      evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::ElementLogical>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
      fd::Tags::Reconstructor, domain::Tags::MeshVelocity<3>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>, ::Tags::Time,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::ElementMap<3, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                  Frame::Inertial>,
      typename System::primitive_variables_tag,
      ::Tags::AnalyticSolution<SolutionForTest>>>(
      EvolutionMetaVars{}, std::move(domain), std::move(boundary_conditions),
      subcell_mesh, subcell_logical_coords, neighbor_data,
      std::unique_ptr<fd::Reconstructor>{
          std::make_unique<ReconstructorForTest>()},
      volume_mesh_velocity, normal_vectors, time,
      clone_unique_ptrs(functions_of_time),
      ElementMap<3, Frame::Grid>{
          ElementId<3>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<3>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}),
      volume_prim_vars, solution);

  // compute FD ghost data and retrieve the result
  fd::BoundaryConditionGhostData::apply(make_not_null(&box), element,
                                        ReconstructorForTest{});
  const auto direction = Direction<3>::upper_xi();
  const std::pair mortar_id = {direction, ElementId<3>::external_boundary_id()};
  const DataVector& fd_ghost_data =
      get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box)
          .at(mortar_id)
          .neighbor_ghost_data_for_reconstruction();

  // Copy the computed FD ghost data into a Variables object in order to
  // facilitate comparison. Note that the returned FD ghost data contains the
  // product of the lorentz factor and spatial velocity, Wv^i, not each of
  // those separately.
  using LorentzFactorTimesSpatialVelocity =
      hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
  using prims_to_reconstruct = grmhd::GhValenciaDivClean::Tags::
      primitive_grmhd_and_spacetime_reconstruction_tags;
  const size_t num_face_pts{
      subcell_mesh.extents().slice_away(direction.dimension()).product()};
  Variables<prims_to_reconstruct> fd_ghost_vars{ghost_zone_size * num_face_pts};
  std::copy(fd_ghost_data.begin(), fd_ghost_data.end(), fd_ghost_vars.data());

  //
  // now test each boundary conditions
  //
  if (typeid(BoundaryConditionType) ==
      typeid(BoundaryConditions::DirichletAnalytic)) {
    const auto ghost_logical_coords =
        evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
            subcell_mesh, ghost_zone_size, direction);
    const auto ghost_inertial_coords = (*grid_to_inertial_map)(
        logical_to_grid_map(ghost_logical_coords), time, functions_of_time);

    const auto [expected_rest_mass_density, expected_electron_fraction,
                expected_pressure, expected_lorentz_factor,
                expected_spatial_velocity, expected_magnetic_field,
                expected_div_cleaning_field, expected_spacetime_metric,
                expected_pi, expected_phi] =
        solution.variables(
            ghost_inertial_coords, time,
            tmpl::list<RestMassDensity, ElectronFraction, Pressure,
                       LorentzFactor, SpatialVelocity, MagneticField,
                       DivergenceCleaningField, SpacetimeMetric, Pi, Phi>{});

    // check values
    CHECK_ITERABLE_APPROX(get<RestMassDensity>(fd_ghost_vars),
                          expected_rest_mass_density);
    CHECK_ITERABLE_APPROX(get<ElectronFraction>(fd_ghost_vars),
                          expected_electron_fraction);
    CHECK_ITERABLE_APPROX(get<Pressure>(fd_ghost_vars), expected_pressure);
    {
      tnsr::I<DataVector, 3> expected_lorentz_factor_times_spatial_velocity{
          ghost_zone_size * num_face_pts};
      for (size_t i = 0; i < 3; ++i) {
        expected_lorentz_factor_times_spatial_velocity.get(i) =
            get(expected_lorentz_factor) * expected_spatial_velocity.get(i);
      }
      CHECK_ITERABLE_APPROX(
          get<LorentzFactorTimesSpatialVelocity>(fd_ghost_vars),
          expected_lorentz_factor_times_spatial_velocity);
    }
    CHECK_ITERABLE_APPROX(get<MagneticField>(fd_ghost_vars),
                          expected_magnetic_field);
    CHECK_ITERABLE_APPROX(get<DivergenceCleaningField>(fd_ghost_vars),
                          expected_div_cleaning_field);
    CHECK_ITERABLE_APPROX(get<SpacetimeMetric>(fd_ghost_vars),
                          expected_spacetime_metric);
    CHECK_ITERABLE_APPROX(get<Pi>(fd_ghost_vars), expected_pi);
    CHECK_ITERABLE_APPROX(get<Phi>(fd_ghost_vars), expected_phi);
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.BCondGhostData",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/"};

  test(grmhd::GhValenciaDivClean::BoundaryConditions::DirichletAnalytic{});
  CHECK_THROWS_WITH(test(grmhd::GhValenciaDivClean::BoundaryConditions::
                             ConstraintPreservingFreeOutflow{}),
                    Catch::Contains("Not implemented because it's not trivial "
                                    "to figure out what the right way of"));

// check that the periodic BC fails
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(test(domain::BoundaryConditions::Periodic<
                         BoundaryConditions::BoundaryCondition>{}),
                    Catch::Contains("not on external boundaries"));
#endif
}
}  // namespace
}  // namespace grmhd::GhValenciaDivClean
