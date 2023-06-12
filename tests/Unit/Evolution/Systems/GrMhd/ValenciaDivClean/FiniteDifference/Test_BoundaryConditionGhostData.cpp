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

#include "DataStructures/TaggedContainers.hpp"
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
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean {
namespace {

// Metavariables to parse the list of derived classes of boundary conditions
struct EvolutionMetaVars {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BoundaryConditions::BoundaryCondition,
                             BoundaryConditions::standard_boundary_conditions>>;
  };
};

template <typename T>
using Flux = ::Tags::Flux<T, tmpl::size_t<3>, Frame::Inertial>;

template <typename BoundaryConditionType>
void test(const BoundaryConditionType& boundary_condition,
          const bool set_fluxes) {
  CAPTURE(set_fluxes);
  const size_t num_dg_pts = 3;

  // Create a 3D element [-1, 1]^3 and use it for test
  const std::array<double, 3> lower_bounds{-1.0, -1.0, -1.0};
  const std::array<double, 3> upper_bounds{1.0, 1.0, 1.0};
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

  using SolutionForTest = Solutions::SmoothFlow;
  const std::array<double, 3> mean_velocity = {{0.23, 0.01, 0.31}};
  const std::array<double, 3> wave_vector = {{0.11, 0.23, 0.32}};
  const double pressure = 1.0;
  const double adiabatic_index = 1.4;
  const double perturbation_size = 0.3;
  const SolutionForTest solution{mean_velocity, wave_vector, pressure,
                                 adiabatic_index, perturbation_size};

  // Below are tags used for testing DemandOutgoingCharSpeeds boundary condition
  //  - volume metric and primitive variables on subcell mesh
  //  - mesh velocity
  //  - normal vectors

  const auto subcell_inertial_coords = (*grid_to_inertial_map)(
      logical_to_grid_map(subcell_logical_coords), time, functions_of_time);

  using RestMassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using ElectronFraction = hydro::Tags::ElectronFraction<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using LorentzFactor = hydro::Tags::LorentzFactor<DataVector>;
  using SpatialVelocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;
  using SpecificInternalEnergy =
      hydro::Tags::SpecificInternalEnergy<DataVector>;
  using SpecificEnthalpy = hydro::Tags::SpecificEnthalpy<DataVector>;

  // Use the Minkowski spacetime for spacetime vars, but we manually tweak
  // values of shift vector so that the DemandOutgoingCharSpeeds condition can
  // be satisfied at the first place. Later we will change shift vector to its
  // original value (0.0) and check that the DemandOutgoingCharSpeeds condition
  // is violated as expected.
  Variables<typename System::spacetime_variables_tag::tags_list>
      volume_spacetime_vars{subcell_mesh.number_of_grid_points()};
  volume_spacetime_vars.assign_subset(solution.variables(
      subcell_inertial_coords, time,
      typename System::spacetime_variables_tag::tags_list{}));
  for (size_t i = 0; i < 3; ++i) {
    get<gr::Tags::Shift<DataVector, 3>>(volume_spacetime_vars).get(i) = -2.0;
  }

  Variables<typename System::primitive_variables_tag::tags_list>
      volume_prim_vars{subcell_mesh.number_of_grid_points()};

  get(get<RestMassDensity>(volume_prim_vars)) = 1.0;
  get(get<ElectronFraction>(volume_prim_vars)) = 0.1;
  get(get<Pressure>(volume_prim_vars)) = 1.0;
  get(get<LorentzFactor>(volume_prim_vars)) = 2.0;
  get(get<SpecificInternalEnergy>(volume_prim_vars)) = 0.3;
  get(get<SpecificEnthalpy>(volume_prim_vars)) = 1.003;
  for (size_t i = 0; i < 3; ++i) {
    get<SpatialVelocity>(volume_prim_vars).get(i) = 0.1;
    get<MagneticField>(volume_prim_vars).get(i) = 0.5;
  }
  get(get<DivergenceCleaningField>(volume_prim_vars)) = 1e-2;

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

  // Compute the cell-centered fluxes.
  using cons_tags = typename System::variables_tag::tags_list;
  typename System::variables_tag::type volume_cons_vars{
      subcell_mesh.number_of_grid_points()};
  typename evolution::dg::subcell::Tags::CellCenteredFlux<
      typename System::flux_variables, 3>::type cell_centered_fluxes{};
  if (set_fluxes) {
    cell_centered_fluxes = typename decltype(cell_centered_fluxes)::value_type{
        subcell_mesh.number_of_grid_points()};
    ConservativeFromPrimitive::apply(
        get<tmpl::at_c<cons_tags, 0>>(make_not_null(&volume_cons_vars)),
        get<tmpl::at_c<cons_tags, 1>>(make_not_null(&volume_cons_vars)),
        get<tmpl::at_c<cons_tags, 2>>(make_not_null(&volume_cons_vars)),
        get<tmpl::at_c<cons_tags, 3>>(make_not_null(&volume_cons_vars)),
        get<tmpl::at_c<cons_tags, 4>>(make_not_null(&volume_cons_vars)),
        get<tmpl::at_c<cons_tags, 5>>(make_not_null(&volume_cons_vars)),
        get<hydro::Tags::RestMassDensity<DataVector>>(volume_prim_vars),
        get<hydro::Tags::ElectronFraction<DataVector>>(volume_prim_vars),
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(volume_prim_vars),
        get<hydro::Tags::Pressure<DataVector>>(volume_prim_vars),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(volume_prim_vars),
        get<hydro::Tags::LorentzFactor<DataVector>>(volume_prim_vars),
        get<hydro::Tags::MagneticField<DataVector, 3>>(volume_prim_vars),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(volume_spacetime_vars),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(volume_spacetime_vars),
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(
            volume_prim_vars));

    ComputeFluxes::apply(
        get<Flux<tmpl::at_c<cons_tags, 0>>>(
            make_not_null(&cell_centered_fluxes.value())),
        get<Flux<tmpl::at_c<cons_tags, 1>>>(
            make_not_null(&cell_centered_fluxes.value())),
        get<Flux<tmpl::at_c<cons_tags, 2>>>(
            make_not_null(&cell_centered_fluxes.value())),
        get<Flux<tmpl::at_c<cons_tags, 3>>>(
            make_not_null(&cell_centered_fluxes.value())),
        get<Flux<tmpl::at_c<cons_tags, 4>>>(
            make_not_null(&cell_centered_fluxes.value())),
        get<Flux<tmpl::at_c<cons_tags, 5>>>(
            make_not_null(&cell_centered_fluxes.value())),

        get<tmpl::at_c<cons_tags, 0>>(volume_cons_vars),
        get<tmpl::at_c<cons_tags, 1>>(volume_cons_vars),
        get<tmpl::at_c<cons_tags, 2>>(volume_cons_vars),
        get<tmpl::at_c<cons_tags, 3>>(volume_cons_vars),
        get<tmpl::at_c<cons_tags, 4>>(volume_cons_vars),
        get<tmpl::at_c<cons_tags, 5>>(volume_cons_vars),

        get<gr::Tags::Lapse<DataVector>>(volume_spacetime_vars),
        get<gr::Tags::Shift<DataVector, 3>>(volume_spacetime_vars),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(volume_spacetime_vars),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(volume_spacetime_vars),
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
            volume_spacetime_vars),
        get<hydro::Tags::Pressure<DataVector>>(volume_prim_vars),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(volume_prim_vars),
        get<hydro::Tags::LorentzFactor<DataVector>>(volume_prim_vars),
        get<hydro::Tags::MagneticField<DataVector, 3>>(volume_prim_vars));
  }

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
      typename System::spacetime_variables_tag,
      typename System::primitive_variables_tag,
      ::Tags::AnalyticSolution<SolutionForTest>,
      evolution::dg::subcell::Tags::CellCenteredFlux<
          typename System::flux_variables, 3>>>(
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
      volume_spacetime_vars, volume_prim_vars, solution, cell_centered_fluxes);

  {
    // compute FD ghost data and retrieve the result
    fd::BoundaryConditionGhostData::apply(make_not_null(&box), element,
                                          ReconstructorForTest{});
    const auto direction = Direction<3>::upper_xi();
    const std::pair mortar_id = {direction,
                                 ElementId<3>::external_boundary_id()};
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
    using prims_to_reconstruct =
        tmpl::list<RestMassDensity, ElectronFraction, Pressure,
                   LorentzFactorTimesSpatialVelocity, MagneticField,
                   DivergenceCleaningField>;
    const size_t num_face_pts{
        subcell_mesh.extents().slice_away(direction.dimension()).product()};
    Variables<prims_to_reconstruct> fd_ghost_vars{ghost_zone_size *
                                                  num_face_pts};
    std::copy(fd_ghost_data.begin(),
              std::next(fd_ghost_data.begin(),
                        static_cast<std::ptrdiff_t>(fd_ghost_vars.size())),
              fd_ghost_vars.data());
    using VarsFluxes =
        Variables<db::wrap_tags_in<Flux, typename System::flux_variables>>;
    VarsFluxes fd_ghost_fluxes{ghost_zone_size * num_face_pts, 0.0};
    if (set_fluxes) {
      REQUIRE(fd_ghost_data.size() ==
              (fd_ghost_vars.size() + fd_ghost_fluxes.size()));
      std::copy(std::next(fd_ghost_data.begin(),
                          static_cast<std::ptrdiff_t>(fd_ghost_vars.size())),
                fd_ghost_data.end(), fd_ghost_fluxes.data());
    } else {
      REQUIRE(fd_ghost_data.size() == fd_ghost_vars.size());
    }

    const auto check_fluxes =
        [&fd_ghost_fluxes](
            const auto& rest_mass_density, const auto& electron_fraction,
            const auto& specific_internal_energy, const auto& local_pressure,
            const auto& spatial_velocity, const auto& lorentz_factor,
            const auto& magnetic_field, const auto& div_cleaning_field,
            const double shift_value) {
          typename System::variables_tag::type expected_cons_vars{
              get(rest_mass_density).size()};
          tnsr::ii<DataVector, 3> spatial_metric(get(rest_mass_density).size(),
                                                 0.0);
          tnsr::II<DataVector, 3> inverse_spatial_metric(
              get(rest_mass_density).size(), 0.0);
          for (size_t i = 0; i < 3; ++i) {
            spatial_metric.get(i, i) = 1.0;
            inverse_spatial_metric.get(i, i) = 1.0;
          }
          const Scalar<DataVector> sqrt_det_spatial_metric(
              get(rest_mass_density).size(), 1.0);
          const Scalar<DataVector> lapse(get(rest_mass_density).size(), 1.0);
          const tnsr::I<DataVector, 3> shift(get(rest_mass_density).size(),
                                             shift_value);
          typename evolution::dg::subcell::Tags::CellCenteredFlux<
              typename System::flux_variables, 3>::type::value_type
              expected_neighbor_fluxes{get(rest_mass_density).size()};
          ConservativeFromPrimitive::apply(
              get<tmpl::at_c<cons_tags, 0>>(make_not_null(&expected_cons_vars)),
              get<tmpl::at_c<cons_tags, 1>>(make_not_null(&expected_cons_vars)),
              get<tmpl::at_c<cons_tags, 2>>(make_not_null(&expected_cons_vars)),
              get<tmpl::at_c<cons_tags, 3>>(make_not_null(&expected_cons_vars)),
              get<tmpl::at_c<cons_tags, 4>>(make_not_null(&expected_cons_vars)),
              get<tmpl::at_c<cons_tags, 5>>(make_not_null(&expected_cons_vars)),
              rest_mass_density, electron_fraction, specific_internal_energy,
              local_pressure, spatial_velocity, lorentz_factor, magnetic_field,
              sqrt_det_spatial_metric, spatial_metric, div_cleaning_field);

          ComputeFluxes::apply(
              get<Flux<tmpl::at_c<cons_tags, 0>>>(
                  make_not_null(&expected_neighbor_fluxes)),
              get<Flux<tmpl::at_c<cons_tags, 1>>>(
                  make_not_null(&expected_neighbor_fluxes)),
              get<Flux<tmpl::at_c<cons_tags, 2>>>(
                  make_not_null(&expected_neighbor_fluxes)),
              get<Flux<tmpl::at_c<cons_tags, 3>>>(
                  make_not_null(&expected_neighbor_fluxes)),
              get<Flux<tmpl::at_c<cons_tags, 4>>>(
                  make_not_null(&expected_neighbor_fluxes)),
              get<Flux<tmpl::at_c<cons_tags, 5>>>(
                  make_not_null(&expected_neighbor_fluxes)),

              get<tmpl::at_c<cons_tags, 0>>(expected_cons_vars),
              get<tmpl::at_c<cons_tags, 1>>(expected_cons_vars),
              get<tmpl::at_c<cons_tags, 2>>(expected_cons_vars),
              get<tmpl::at_c<cons_tags, 3>>(expected_cons_vars),
              get<tmpl::at_c<cons_tags, 4>>(expected_cons_vars),
              get<tmpl::at_c<cons_tags, 5>>(expected_cons_vars),

              lapse, shift, sqrt_det_spatial_metric, spatial_metric,
              inverse_spatial_metric, local_pressure, spatial_velocity,
              lorentz_factor, magnetic_field);
          tmpl::for_each<tmpl::list<tmpl::at_c<cons_tags, 0>>>(
              [&fd_ghost_fluxes, &expected_neighbor_fluxes](auto cons_tag_v) {
                using tag =
                    Flux<tmpl::type_from<std::decay_t<decltype(cons_tag_v)>>>;
                CAPTURE(pretty_type::get_name<tag>());
                CHECK_ITERABLE_APPROX(get<tag>(fd_ghost_fluxes),
                                      get<tag>(expected_neighbor_fluxes));
              });
        };

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

      const auto rest_mass_density_py = pypp::call<Scalar<DataVector>>(
          "GrMhd.SmoothFlow", "rest_mass_density", ghost_inertial_coords, time,
          mean_velocity, wave_vector, pressure, adiabatic_index,
          perturbation_size);
      const auto electron_fraction_py = pypp::call<Scalar<DataVector>>(
          "GrMhd.SmoothFlow", "electron_fraction", ghost_inertial_coords, time,
          mean_velocity, wave_vector, pressure, adiabatic_index,
          perturbation_size);
      const auto pressure_py = pypp::call<Scalar<DataVector>>(
          "GrMhd.SmoothFlow", "pressure", ghost_inertial_coords, time,
          mean_velocity, wave_vector, pressure, adiabatic_index,
          perturbation_size);
      const auto specific_internal_energy_py = pypp::call<Scalar<DataVector>>(
          "GrMhd.SmoothFlow", "specific_internal_energy", ghost_inertial_coords,
          time, mean_velocity, wave_vector, pressure, adiabatic_index,
          perturbation_size);
      const auto lorentz_factor_py = pypp::call<Scalar<DataVector>>(
          "GrMhd.SmoothFlow", "lorentz_factor", ghost_inertial_coords, time,
          mean_velocity, wave_vector, pressure, adiabatic_index,
          perturbation_size);
      const auto spatial_velocity_py = pypp::call<tnsr::I<DataVector, 3>>(
          "GrMhd.SmoothFlow", "spatial_velocity", ghost_inertial_coords, time,
          mean_velocity, wave_vector, pressure, adiabatic_index,
          perturbation_size);
      const auto magnetic_field_py = pypp::call<tnsr::I<DataVector, 3>>(
          "GrMhd.SmoothFlow", "magnetic_field", ghost_inertial_coords, time,
          mean_velocity, wave_vector, pressure, adiabatic_index,
          perturbation_size);
      const auto div_cleaning_field_py = pypp::call<Scalar<DataVector>>(
          "GrMhd.SmoothFlow", "divergence_cleaning_field",
          ghost_inertial_coords, time, mean_velocity, wave_vector, pressure,
          adiabatic_index, perturbation_size);

      // check values
      CHECK_ITERABLE_APPROX(get<RestMassDensity>(fd_ghost_vars),
                            rest_mass_density_py);
      CHECK_ITERABLE_APPROX(get<ElectronFraction>(fd_ghost_vars),
                            electron_fraction_py);
      CHECK_ITERABLE_APPROX(get<Pressure>(fd_ghost_vars), pressure_py);
      {
        tnsr::I<DataVector, 3> lorentz_factor_times_spatial_velocity_py{
            ghost_zone_size * num_face_pts};
        for (size_t i = 0; i < 3; ++i) {
          lorentz_factor_times_spatial_velocity_py.get(i) =
              get(lorentz_factor_py) * spatial_velocity_py.get(i);
        }
        CHECK_ITERABLE_APPROX(
            get<LorentzFactorTimesSpatialVelocity>(fd_ghost_vars),
            lorentz_factor_times_spatial_velocity_py);
      }
      CHECK_ITERABLE_APPROX(get<MagneticField>(fd_ghost_vars),
                            magnetic_field_py);
      CHECK_ITERABLE_APPROX(get<DivergenceCleaningField>(fd_ghost_vars),
                            div_cleaning_field_py);
      if (set_fluxes) {
        check_fluxes(rest_mass_density_py, electron_fraction_py,
                     specific_internal_energy_py, pressure_py,
                     spatial_velocity_py, lorentz_factor_py, magnetic_field_py,
                     div_cleaning_field_py, 0.0);
      }
    }

    if (typeid(BoundaryConditionType) ==
        typeid(BoundaryConditions::DemandOutgoingCharSpeeds)) {
      // At this moment shift vector was set to be that the
      // DemandOutgoingCharSpeedscondition does not throw any error. Here we
      // just check if `fd::BoundaryConditionGhostData::apply()` has correctly
      // filled out `fd_ghost_data` with the outermost value.
      Variables<prims_to_reconstruct> expected_ghost_vars{ghost_zone_size *
                                                          num_face_pts};
      get(get<RestMassDensity>(expected_ghost_vars)) = 1.0;
      get(get<Pressure>(expected_ghost_vars)) = 1.0;
      get(get<ElectronFraction>(expected_ghost_vars)) = 0.1;
      for (size_t i = 0; i < 3; ++i) {
        get<LorentzFactorTimesSpatialVelocity>(expected_ghost_vars).get(i) =
            0.2;
        get<MagneticField>(expected_ghost_vars).get(i) = 0.5;
      }
      get(get<DivergenceCleaningField>(expected_ghost_vars)) = 1e-2;

      tmpl::for_each<prims_to_reconstruct>(
          [&expected_ghost_vars, &fd_ghost_vars](auto tag_v) {
            using tag = typename tmpl::type_from<decltype(tag_v)>;
            CHECK_ITERABLE_APPROX(get<tag>(expected_ghost_vars),
                                  get<tag>(fd_ghost_vars));
          });

      if (set_fluxes) {
        const Scalar<DataVector> expected_specific_internal_energy(
            expected_ghost_vars.number_of_grid_points(), 0.3);
        const Scalar<DataVector> expected_lorentz_factor(
            expected_ghost_vars.number_of_grid_points(),
            sqrt(1.0 +
                 square(get<0>(get<LorentzFactorTimesSpatialVelocity>(
                     expected_ghost_vars))[0]) +
                 square(get<1>(get<LorentzFactorTimesSpatialVelocity>(
                     expected_ghost_vars))[0]) +
                 square(get<2>(get<LorentzFactorTimesSpatialVelocity>(
                     expected_ghost_vars))[0])));
        const tnsr::I<DataVector, 3> expected_spatial_velocity =
            tenex::evaluate<ti::I>(get<LorentzFactorTimesSpatialVelocity>(
                                       expected_ghost_vars)(ti::I) /
                                   expected_lorentz_factor());
        check_fluxes(get<RestMassDensity>(expected_ghost_vars),
                     get<ElectronFraction>(expected_ghost_vars),
                     expected_specific_internal_energy,
                     get<Pressure>(expected_ghost_vars),
                     expected_spatial_velocity, expected_lorentz_factor,
                     get<MagneticField>(expected_ghost_vars),
                     get<DivergenceCleaningField>(expected_ghost_vars), -2.0);
      }

      // Set shift to be zero so that the DemandOutgoingCharSpeeds condition is
      // violated. See if the code fails correctly.
      db::mutate<gr::Tags::Shift<DataVector, 3>>(
          [](const gsl::not_null<tnsr::I<DataVector, 3>*> shift_vector) {
            for (size_t i = 0; i < 3; ++i) {
              (*shift_vector).get(i) = 0.0;
            }
          },
          make_not_null(&box));
      CHECK_THROWS_WITH(
          ([&box, &element]() {
            fd::BoundaryConditionGhostData::apply(make_not_null(&box), element,
                                                  ReconstructorForTest{});
          })(),
          Catch::Contains(
              "Subcell DemandOutgoingCharSpeeds boundary condition violated"));

      // Test when the volume mesh velocity has value, which will raise ERROR.
      // See if the code fails correctly.
      db::mutate<domain::Tags::MeshVelocity<3>>(
          [&subcell_mesh](
              const gsl::not_null<std::optional<tnsr::I<DataVector, 3>>*>
                  mesh_velocity) {
            tnsr::I<DataVector, 3> volume_mesh_velocity_tnsr(
                subcell_mesh.number_of_grid_points(), 0.0);
            *mesh_velocity = volume_mesh_velocity_tnsr;
          },
          make_not_null(&box));
      CHECK_THROWS_WITH(
          ([&box, &element]() {
            fd::BoundaryConditionGhostData::apply(make_not_null(&box), element,
                                                  ReconstructorForTest{});
          })(),
          Catch::Contains("Subcell currently does not support moving mesh"));
    }

    if (typeid(BoundaryConditionType) ==
        typeid(BoundaryConditions::HydroFreeOutflow)) {
      Variables<prims_to_reconstruct> expected_ghost_vars{ghost_zone_size *
                                                          num_face_pts};
      // cf) See line 172-179 for values of prim variables in volume.
      get(get<RestMassDensity>(expected_ghost_vars)) = 1.0;
      get(get<Pressure>(expected_ghost_vars)) = 1.0;
      get(get<ElectronFraction>(expected_ghost_vars)) = 0.1;
      for (size_t i = 0; i < 3; ++i) {
        get<LorentzFactorTimesSpatialVelocity>(expected_ghost_vars).get(i) =
            0.2;
        get<MagneticField>(expected_ghost_vars).get(i) = 0.5;
      }
      get(get<DivergenceCleaningField>(expected_ghost_vars)) =
          0.0;  // note the difference from DemandOutgoingCharSpeeds boundary
                // condition. While the volume value was set to 1e-2,
                // HydroFreeOutflow boundary should assign zero to Phi in ghost
                // zones.

      tmpl::for_each<prims_to_reconstruct>(
          [&expected_ghost_vars, &fd_ghost_vars](auto tag_v) {
            using tag = typename tmpl::type_from<decltype(tag_v)>;
            CHECK_ITERABLE_APPROX(get<tag>(expected_ghost_vars),
                                  get<tag>(fd_ghost_vars));
          });
      if (set_fluxes) {
        const Scalar<DataVector> expected_specific_internal_energy(
            expected_ghost_vars.number_of_grid_points(), 0.3);
        const Scalar<DataVector> expected_lorentz_factor(
            expected_ghost_vars.number_of_grid_points(),
            sqrt(1.0 +
                 square(get<0>(get<LorentzFactorTimesSpatialVelocity>(
                     expected_ghost_vars))[0]) +
                 square(get<1>(get<LorentzFactorTimesSpatialVelocity>(
                     expected_ghost_vars))[0]) +
                 square(get<2>(get<LorentzFactorTimesSpatialVelocity>(
                     expected_ghost_vars))[0])));
        const tnsr::I<DataVector, 3> expected_spatial_velocity =
            tenex::evaluate<ti::I>(get<LorentzFactorTimesSpatialVelocity>(
                                       expected_ghost_vars)(ti::I) /
                                   expected_lorentz_factor());
        check_fluxes(get<RestMassDensity>(expected_ghost_vars),
                     get<ElectronFraction>(expected_ghost_vars),
                     expected_specific_internal_energy,
                     get<Pressure>(expected_ghost_vars),
                     expected_spatial_velocity, expected_lorentz_factor,
                     get<MagneticField>(expected_ghost_vars),
                     get<DivergenceCleaningField>(expected_ghost_vars), -2.0);
      }

      // Now set the spatial velocity pointing inward with respect to the
      // upper_xi() direction which is the direction with which test is done.
      // Then re-compute ghost zone data to see if inward-pointing components
      // are set to zero.
      db::mutate<SpatialVelocity>(
          [](const gsl::not_null<tnsr::I<DataVector, 3>*>
                 interior_spatial_velocity) {
            (*interior_spatial_velocity).get(0) = -0.2;
          },
          make_not_null(&box));
      fd::BoundaryConditionGhostData::apply(make_not_null(&box), element,
                                            ReconstructorForTest{});

      const DataVector& fd_ghost_data_velocity_inward =
          get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box)
              .at(mortar_id)
              .neighbor_ghost_data_for_reconstruction();
      Variables<prims_to_reconstruct> fd_ghost_vars_velocity_inward{
          ghost_zone_size * num_face_pts};
      std::copy(fd_ghost_data_velocity_inward.begin(),
                std::next(fd_ghost_data_velocity_inward.begin(),
                          static_cast<std::ptrdiff_t>(
                              fd_ghost_vars_velocity_inward.size())),
                fd_ghost_vars_velocity_inward.data());

      // we expect Wv^0 is zero
      get<LorentzFactorTimesSpatialVelocity>(expected_ghost_vars).get(0) = 0.0;

      tmpl::for_each<prims_to_reconstruct>(
          [&expected_ghost_vars, &fd_ghost_vars_velocity_inward](auto tag_v) {
            using tag = typename tmpl::type_from<decltype(tag_v)>;
            CHECK_ITERABLE_APPROX(get<tag>(expected_ghost_vars),
                                  get<tag>(fd_ghost_vars_velocity_inward));
          });
    }
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Fd.BCondGhostData",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/"};

  for (const bool set_fluxes : {true, false}) {
    test(BoundaryConditions::DirichletAnalytic{}, set_fluxes);
    test(BoundaryConditions::DemandOutgoingCharSpeeds{}, set_fluxes);
    test(BoundaryConditions::HydroFreeOutflow{}, set_fluxes);
  }

// check that the periodic BC fails
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(([]() {
                      test(domain::BoundaryConditions::Periodic<
                               BoundaryConditions::BoundaryCondition>{},
                           false);
                    })(),
                    Catch::Contains("not on external boundaries"));
#endif
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean
