// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TimeDerivativeTerms.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TransportVelocity.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"

namespace {

// Wrapper fixing Mesh to non-cartoon so check_with_random_values can test
// the standard source terms against the Python functions.
void apply_standard_sources(
    const gsl::not_null<Scalar<DataVector>*> source_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        source_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> source_tilde_phi,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& pressure, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const double constraint_damping_parameter) {
  const size_t num_pts = get(tilde_d).size();
  const Mesh<3> mesh{3_st, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const tnsr::I<DataVector, 3, Frame::Inertial> dg_inertial_coords{
      num_pts, std::numeric_limits<double>::signaling_NaN()};  // unused
  grmhd::ValenciaDivClean::ComputeSources::apply(
      source_tilde_tau, source_tilde_s, source_tilde_b, source_tilde_phi,
      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,
      spatial_velocity, magnetic_field, rest_mass_density, electron_fraction,
      specific_internal_energy, lorentz_factor, pressure, lapse, shift, d_lapse,
      d_shift, spatial_metric, d_spatial_metric, inv_spatial_metric,
      sqrt_det_spatial_metric, extrinsic_curvature,
      constraint_damping_parameter, dg_inertial_coords, mesh);
}

// Metavariables needed for FD subcell time derivative test
struct DummyEvolutionMetaVars {
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = true;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            grmhd::ValenciaDivClean::BoundaryConditions::BoundaryCondition,
            grmhd::ValenciaDivClean::BoundaryConditions::
                standard_boundary_conditions>,
        tmpl::pair<evolution::BoundaryCorrection,
                   grmhd::ValenciaDivClean::BoundaryCorrections::
                       standard_boundary_corrections>>;
  };
};

struct DummyAnalyticSolutionTag : db::SimpleTag {
  using type = grmhd::Solutions::BondiMichel;
};

struct HydroResults {
  Scalar<DataVector> dt_tilde_d;
  Scalar<DataVector> dt_tilde_ye;
  Scalar<DataVector> dt_tilde_tau;
  tnsr::i<DataVector, 3> dt_tilde_s;
  tnsr::I<DataVector, 3> dt_tilde_b;
  Scalar<DataVector> dt_tilde_phi;
};

// Evaluates the RHS for a full 3D evolution and a cartoon evolution and
// checks they are the same (requires extra cartoon source terms as the
// "manual" add_cartesian_flux_divergence does not know about off-diagonal
// terms that cartoon requires)
template <bool Spherical>
void test_cartoon_fd_time_derivative() {
  CAPTURE(Spherical);
  const double time = 0.0;
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      fot{};

  const grmhd::Solutions::BondiMichel soln{1.0, 5.0, 0.05, 1.4, 2.0};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3d =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  constexpr size_t Nx_dg = 20;
  constexpr Spectral::Quadrature quad_c =
      Spherical ? Spectral::Quadrature::SphericalSymmetry
                : Spectral::Quadrature::AxialSymmetry;
  const Mesh<3> dg_mesh_c{
      {Nx_dg, Spherical ? 1_st : Nx_dg, 1_st},
      {{Spectral::Basis::Legendre,
        Spherical ? Spectral::Basis::Cartoon : Spectral::Basis::Legendre,
        Spectral::Basis::Cartoon}},
      {{Spectral::Quadrature::GaussLobatto,
        Spherical ? quad_c : Spectral::Quadrature::GaussLobatto, quad_c}}};
  const Mesh<3> dg_mesh_3{Nx_dg, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  // lambda to compute du/dt of conserved variables and put in a HydroResults
  // struct for comparison between full 3D and cartoon timestep
  const auto fd_compute = [&soln, time, &fot](const Mesh<3>& dg_mesh) {
    const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
    const size_t comp_dim =
        subcell_mesh.basis(1) == Spectral::Basis::Cartoon
            ? 1
            : (subcell_mesh.basis(2) == Spectral::Basis::Cartoon ? 2 : 3);
    const auto cartoon_neighbors =
        comp_dim == 3
            ? std::unordered_set<Direction<3>>{}
            : (comp_dim == 2
                   ? std::unordered_set<
                         Direction<3>>{Direction<3>::lower_zeta(),
                                       Direction<3>::upper_zeta()}
                   : std::unordered_set<Direction<3>>{
                         Direction<3>::lower_eta(), Direction<3>::upper_eta(),
                         Direction<3>::lower_zeta(),
                         Direction<3>::upper_zeta()});

    std::vector<Block<3>> blocks;
    blocks.emplace_back(Block<3>(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Affine3d{Affine{-1., 1., 2.0, 2.5}, Affine{-1., 1., -0.25, 0.25},
                     Affine{-1., 1., -0.05, 0.05}}),
        0, {}, "",
        comp_dim == 1 ? domain::topologies::cartoon_sphere
                      : (comp_dim == 2 ? domain::topologies::cartoon_cylinder
                                       : domain::topologies::hypercube<3>)));

    // NOLINTNEXTLINE(misc-const-correctness)
    Domain<3> domain{std::move(blocks)};

    // Create DirichletAnalytic boundary condition
    auto dirichlet_bc = std::make_unique<
        grmhd::ValenciaDivClean::BoundaryConditions::DirichletAnalytic>(
        std::make_unique<grmhd::Solutions::BondiMichel>(soln));

    // Create boundary conditions only for non-cartoon directions
    DirectionMap<3,
                 std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
        block_boundary_conditions;

    for (const auto& direction : Direction<3>::all_directions()) {
      if (cartoon_neighbors.contains(direction)) {
        continue;
      }
      block_boundary_conditions[direction] = dirichlet_bc->get_clone();
    }

    std::vector<DirectionMap<
        3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        boundary_conditions_vector;
    boundary_conditions_vector.push_back(std::move(block_boundary_conditions));

    const auto element = domain::create_initial_element(
        ElementId<3>{0, {SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}},
        domain.blocks(),
        std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{0, 0, 0}}});

    const ElementMap<3, Frame::Grid> logical_to_grid_map{element.id(),
                                                         domain.blocks()[0]};

    // For simplicity, use Identity for grid-to-inertial since our domain map
    // goes directly to inertial
    const auto grid_to_inertial_map_ptr =
        domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Identity<3>{});

    // Cartoon dimensions are skipped in reconstruction since they have extent 1
    const grmhd::ValenciaDivClean::fd::MonotonisedCentralPrim recons{true};

    const ElementMap<3, Frame::Inertial> logical_to_inertial_map{
        element.id(), domain.blocks()[0].stationary_map().get_clone()};
    const auto logical_coords = logical_coordinates(subcell_mesh);
    const auto cell_centered_coords = logical_to_inertial_map(logical_coords);

    // Get solution variables at subcell points
    Variables<typename grmhd::ValenciaDivClean::System::
                  spacetime_variables_tag::tags_list>
        cell_centered_spacetime_vars{subcell_mesh.number_of_grid_points()};
    cell_centered_spacetime_vars.assign_subset(
        soln.variables(cell_centered_coords, time,
                       typename grmhd::ValenciaDivClean::System::
                           spacetime_variables_tag::tags_list{}));

    Variables<typename grmhd::ValenciaDivClean::System::
                  primitive_variables_tag::tags_list>
        cell_centered_prim_vars{subcell_mesh.number_of_grid_points()};
    cell_centered_prim_vars.assign_subset(
        soln.variables(cell_centered_coords, time,
                       typename grmhd::ValenciaDivClean::System::
                           primitive_variables_tag::tags_list{}));

    // Apply a Gaussian perturbation so du/dt is non-trivial
    {
      constexpr double eps_perturb = 0.1;
      constexpr double r0_perturb = 2.26;
      constexpr double sigma_perturb = 0.2;
      constexpr double gamma_eos = 1.4;

      auto& rho = get<hydro::Tags::RestMassDensity<DataVector>>(
          cell_centered_prim_vars);
      auto& p = get<hydro::Tags::Pressure<DataVector>>(cell_centered_prim_vars);

      const double K_poly = get(p)[0] / std::pow(get(rho)[0], gamma_eos);

      const size_t npts = subcell_mesh.number_of_grid_points();
      DataVector f(npts);
      const DataVector& xn = get<0>(cell_centered_coords);
      const DataVector& yn = get<1>(cell_centered_coords);
      const DataVector& zn = get<2>(cell_centered_coords);
      const DataVector r = [&xn, &yn, &zn]() -> DataVector {
        if constexpr (Spherical) {
          return sqrt(xn * xn + yn * yn + zn * zn);
        } else {
          (void)yn;
          return sqrt(xn * xn + zn * zn);
        }
      }();
      const DataVector dr = r - r0_perturb;
      f = exp(-dr * dr / (2 * sigma_perturb * sigma_perturb));

      get(rho) *= (1.0 + eps_perturb * f);

      auto& eps_int = get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
          cell_centered_prim_vars);
      auto& phi = get<hydro::Tags::DivergenceCleaningField<DataVector>>(
          cell_centered_prim_vars);
      for (size_t n = 0; n < npts; ++n) {
        get(p)[n] = K_poly * std::pow(get(rho)[n], gamma_eos);
      }
      get(eps_int) = get(p) / (get(rho) * (gamma_eos - 1.0));
      get(phi) += eps_perturb * f;

      // Perturb B^phi to exercise the axial cartoon magnetic field source
      // terms. BondiMichel has B^phi = 0 by spherical symmetry, so without
      // this the relevant axial source terms are identically zero.
      if constexpr (not Spherical) {
        auto& b_field = get<hydro::Tags::MagneticField<DataVector, 3>>(
            cell_centered_prim_vars);
        get<0>(b_field) -= eps_perturb * f * zn / r;
        get<2>(b_field) += eps_perturb * f * xn / r;
      }
    }

    Variables<
        typename grmhd::ValenciaDivClean::System::variables_tag::tags_list>
        cell_centered_cons_vars{subcell_mesh.number_of_grid_points()};
    apply(make_not_null(&cell_centered_cons_vars),
          grmhd::ValenciaDivClean::ConservativeFromPrimitive{},
          cell_centered_spacetime_vars, cell_centered_prim_vars);

    evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
        neighbor_data{};

    auto face_centered_gr_vars = [&]() {
      std::array<typename grmhd::ValenciaDivClean::System::
                     flux_spacetime_variables_tag::type,
                 3>
          result{};
      for (size_t d = 0; d < comp_dim; ++d) {
        Index<3> face_mesh_extents = subcell_mesh.extents();
        face_mesh_extents[d] += 1;

        const auto basis = subcell_mesh.basis();
        auto quadrature = subcell_mesh.quadrature();
        gsl::at(quadrature, d) = Spectral::Quadrature::FaceCentered;
        const Mesh<3> face_centered_mesh{face_mesh_extents.indices(), basis,
                                         quadrature};

        const auto face_centered_logical_coords =
            logical_coordinates(face_centered_mesh);
        const auto face_centered_inertial_coords =
            logical_to_inertial_map(face_centered_logical_coords);

        gsl::at(result, d).initialize(face_mesh_extents.product());
        gsl::at(result, d).assign_subset(
            soln.variables(face_centered_inertial_coords, time,
                           typename grmhd::ValenciaDivClean::System::
                               flux_spacetime_variables_tag::tags_list{}));
      }
      return result;
    }();

    using variables_tag =
        typename grmhd::ValenciaDivClean::System::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
    using CellCenteredFluxesTag =
        evolution::dg::subcell::Tags::CellCenteredFlux<
            typename grmhd::ValenciaDivClean::System::flux_variables, 3>;

    auto box = db::create<
        db::AddSimpleTags<
            domain::Tags::Element<3>, evolution::dg::subcell::Tags::Mesh<3>,
            domain::Tags::Mesh<3>,
            grmhd::ValenciaDivClean::fd::Tags::Reconstructor,
            evolution::Tags::BoundaryCorrection,
            hydro::Tags::GrmhdEquationOfState,
            typename grmhd::ValenciaDivClean::System::spacetime_variables_tag,
            typename grmhd::ValenciaDivClean::System::primitive_variables_tag,
            dt_variables_tag, variables_tag,
            evolution::dg::subcell::Tags::OnSubcellFaces<
                typename grmhd::ValenciaDivClean::System::
                    flux_spacetime_variables_tag,
                3>,
            evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
            grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter,
            evolution::dg::Tags::MortarData<3>,
            domain::Tags::ElementMap<3, Frame::Grid>,
            domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                        Frame::Inertial>,
            domain::Tags::Domain<3>,
            domain::Tags::ExternalBoundaryConditions<3>,
            ::Tags::VariableFixer<::VariableFixing::FixToAtmosphere<3>>,
            domain::Tags::MeshVelocity<3, Frame::Inertial>,
            domain::Tags::DivMeshVelocity,
            evolution::dg::Tags::NormalCovectorAndMagnitude<3>, ::Tags::Time,
            domain::Tags::FunctionsOfTimeInitialize, DummyAnalyticSolutionTag,
            Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars>,
            CellCenteredFluxesTag,
            evolution::dg::subcell::Tags::SubcellOptions<3>,
            evolution::dg::subcell::Tags::ReconstructionOrder<3>>,
        db::AddComputeTags<
            ::domain::Tags::LogicalCoordinates<3>,
            // Compute tags for Frame::Grid quantities
            ::domain::Tags::MappedCoordinates<
                ::domain::Tags::ElementMap<3, Frame::Grid>,
                ::domain::Tags::Coordinates<3, Frame::ElementLogical>>,
            ::domain::Tags::InverseJacobianCompute<
                ::domain::Tags::ElementMap<3, Frame::Grid>,
                ::domain::Tags::Coordinates<3, Frame::ElementLogical>>,
            // Compute tags for Frame::Inertial quantities
            ::domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
                ::domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                              Frame::Inertial>>,
            ::domain::Tags::InertialFromGridCoordinatesCompute<3>,
            ::domain::Tags::ElementToInertialInverseJacobian<3>,
            ::domain::Tags::DetInvJacobianCompute<3, Frame::ElementLogical,
                                                  Frame::Inertial>,

            evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>,
            ::domain::Tags::MappedCoordinates<
                ::domain::Tags::ElementMap<3, Frame::Grid>,
                evolution::dg::subcell::Tags::Coordinates<
                    3, Frame::ElementLogical>,
                evolution::dg::subcell::Tags::Coordinates>,
            evolution::dg::subcell::Tags::InertialCoordinatesCompute<
                ::domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                              Frame::Inertial>>,
            evolution::dg::subcell::fd::Tags::
                InverseJacobianLogicalToGridCompute<
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
                    3>>>(
        element, subcell_mesh, dg_mesh,
        std::unique_ptr<grmhd::ValenciaDivClean::fd::Reconstructor>{
            std::make_unique<std::decay_t<decltype(recons)>>(recons)},
        std::unique_ptr<evolution::BoundaryCorrection>{
            std::make_unique<grmhd::ValenciaDivClean::BoundaryCorrections::Hll>(
                1.0e-30, 1.0e-8)},
        soln.equation_of_state().promote_to_3d_eos(),
        cell_centered_spacetime_vars, cell_centered_prim_vars,
        Variables<typename dt_variables_tag::tags_list>{
            subcell_mesh.number_of_grid_points()},
        cell_centered_cons_vars, face_centered_gr_vars, neighbor_data, 0.0,
        evolution::dg::Tags::MortarData<3>::type{},
        ElementMap<3, Frame::Grid>{element.id(),
                                   logical_to_grid_map.block_map().get_clone()},
        grid_to_inertial_map_ptr->get_clone(), std::move(domain),
        std::move(boundary_conditions_vector),
        ::VariableFixing::FixToAtmosphere<3>(1.0e-15, 1.0e-15, {}, {}, {}),
        std::optional<tnsr::I<DataVector, 3>>{},
        std::optional<Scalar<DataVector>>{},
        typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type{},
        time, clone_unique_ptrs(fot), grmhd::Solutions::BondiMichel{},
        DummyEvolutionMetaVars{},
        std::optional<Variables<db::wrap_tags_in<
            ::Tags::Flux,
            typename grmhd::ValenciaDivClean::System::flux_variables,
            tmpl::size_t<3>, Frame::Inertial>>>{},
        evolution::dg::subcell::SubcellOptions{
            1.0e8, 1_st, 1.0e-4, 1.0e-5, false, false,
            evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
            std::nullopt, ::fd::DerivativeOrder::Two, 2, 2, 2},
        typename evolution::dg::subcell::Tags::ReconstructionOrder<3>::type{});
    db::mutate_apply<grmhd::ValenciaDivClean::ConservativeFromPrimitive>(
        make_not_null(&box));

    // Call the subcell time derivative (this includes cartoon corrections)
    grmhd::ValenciaDivClean::subcell::TimeDerivative::apply(
        make_not_null(&box));

    const auto dt_vars = db::get<dt_variables_tag>(box);

    const auto& dt_tilde_s =
        get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeS<>>>(dt_vars);
    const auto& dt_tilde_b =
        get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeB<>>>(dt_vars);
    const auto& dt_tilde_phi =
        get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildePhi>>(dt_vars);
    const auto& dt_tilde_d =
        get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeD>>(dt_vars);
    const auto& dt_tilde_ye =
        get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeYe>>(dt_vars);
    const auto& dt_tilde_tau =
        get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeTau>>(dt_vars);

    const size_t num_points =
        Spherical ? subcell_mesh.slice_through(0).number_of_grid_points()
                  : subcell_mesh.slice_away(2).number_of_grid_points();
    HydroResults result{
        Scalar<DataVector>(num_points, 0.0),
        Scalar<DataVector>(num_points, 0.0),
        Scalar<DataVector>(num_points, 0.0),
        tnsr::i<DataVector, 3>(num_points, 0.0),
        tnsr::I<DataVector, 3>(num_points, 0.0),
        Scalar<DataVector>(num_points, 0.0),
    };
    if (comp_dim == 3) {
      // Extract the x-strip at the central z=0 (y=0) indices for comparison
      // with cartoon FD
      const size_t nx = subcell_mesh.extents(0);
      const size_t ny = subcell_mesh.extents(1);
      const size_t nz = subcell_mesh.extents(2);
      const size_t mid_z = nz / 2;
      if constexpr (Spherical) {
        // spherical slice
        const size_t mid_y = ny / 2;
        const size_t strip_offset = nx * (mid_y + ny * mid_z);
        for (size_t ix = 0; ix < nx; ++ix) {
          const size_t idx = ix + strip_offset;
          get(result.dt_tilde_d)[ix] = get(dt_tilde_d)[idx];
          get(result.dt_tilde_ye)[ix] = get(dt_tilde_ye)[idx];
          get(result.dt_tilde_tau)[ix] = get(dt_tilde_tau)[idx];
          get(result.dt_tilde_phi)[ix] = get(dt_tilde_phi)[idx];
          for (size_t i = 0; i < 3; ++i) {
            result.dt_tilde_s.get(i)[ix] = dt_tilde_s.get(i)[idx];
            result.dt_tilde_b.get(i)[ix] = dt_tilde_b.get(i)[idx];
          }
        }
      } else {
        // axial slice
        const size_t strip_offset = nx * ny * mid_z;
        for (size_t ix = 0; ix < nx; ++ix) {
          for (size_t iy = 0; iy < ny; ++iy) {
            const size_t idx_3 = ix + nx * iy + strip_offset;
            const size_t idx_c = ix + nx * iy;
            get(result.dt_tilde_d)[idx_c] = get(dt_tilde_d)[idx_3];
            get(result.dt_tilde_ye)[idx_c] = get(dt_tilde_ye)[idx_3];
            get(result.dt_tilde_tau)[idx_c] = get(dt_tilde_tau)[idx_3];
            get(result.dt_tilde_phi)[idx_c] = get(dt_tilde_phi)[idx_3];
            for (size_t i = 0; i < 3; ++i) {
              result.dt_tilde_s.get(i)[idx_c] = dt_tilde_s.get(i)[idx_3];
              result.dt_tilde_b.get(i)[idx_c] = dt_tilde_b.get(i)[idx_3];
            }
          }
        }
      }
    } else {
      // Cartoon case, just copy
      get(result.dt_tilde_d) = get(dt_tilde_d);
      get(result.dt_tilde_ye) = get(dt_tilde_ye);
      get(result.dt_tilde_tau) = get(dt_tilde_tau);
      get(result.dt_tilde_phi) = get(dt_tilde_phi);
      for (size_t i = 0; i < 3; ++i) {
        result.dt_tilde_s.get(i) = dt_tilde_s.get(i);
        result.dt_tilde_b.get(i) = dt_tilde_b.get(i);
      }
    }

    return result;
  };

  const auto fd_c = fd_compute(dg_mesh_c);
  const auto fd_3 = fd_compute(dg_mesh_3);

  // The comparison tolerance is set by the extents of the 3D grid which are
  // cartoon in the cartoon grid. For the axial case, the 3D solution has
  // residual z-dependence (e.g. BondiMichel depends on r = sqrt(x^2+y^2+z^2)
  // rather than the cylindrical radius), so the 3D flux divergence picks up
  // O(z^2/r^2) corrections relative to the cartoon, which evaluates everything
  // at z=0 exactly
  const Approx local_approx = Approx::custom().epsilon(0.01).scale(1e-10);
  const Approx local_approx_loose = Approx::custom().epsilon(0.1).scale(1e-5);

  const auto check_without_border =
      [&local_approx, &local_approx_loose](
          const DataVector& a, const DataVector& b, const std::string& name,
          const bool check_loose = false) {
        INFO(name);
        // There are inflated errors at border due to ghost zone being
        // non-perturbed solution
        const size_t buf = 2;
        const size_t y_length = Spherical ? 1 : 2 * Nx_dg - 1;
        for (size_t i = buf * y_length; i < a.size() - buf * y_length; ++i) {
          if (Spherical or
              (i % y_length >= buf and i % y_length <= y_length - buf - 1)) {
            CAPTURE(i);
            if (abs(a[i]) < 1e-13) {
              CHECK(a[i] == approx(b[i]));
            } else if (check_loose) {
              CHECK(a[i] == local_approx_loose(b[i]));
            } else {
              CHECK(a[i] == local_approx(b[i]));
            }
          }
        }
      };
  check_without_border(get(fd_c.dt_tilde_d), get(fd_3.dt_tilde_d),
                       "dt tilde D");
  check_without_border(get(fd_c.dt_tilde_ye), get(fd_3.dt_tilde_ye),
                       "dt tilde Ye");
  check_without_border(get(fd_c.dt_tilde_tau), get(fd_3.dt_tilde_tau),
                       "dt tilde tau");
  check_without_border(get(fd_c.dt_tilde_phi), get(fd_3.dt_tilde_phi),
                       "dt tilde Phi");
  {
    INFO("x component");
    check_without_border(get<0>(fd_c.dt_tilde_b), get<0>(fd_3.dt_tilde_b),
                         "dt tilde B");
    check_without_border(get<0>(fd_c.dt_tilde_s), get<0>(fd_3.dt_tilde_s),
                         "dt tilde S");
  }
  {
    INFO("y component");
    check_without_border(get<1>(fd_c.dt_tilde_b), get<1>(fd_3.dt_tilde_b),
                         "dt tilde B");
    // \tilde{S}_y requires the approx to have the scale set to 1e-5, which is
    // around the magnitude of the components for both 3D and cartoon cases.
    check_without_border(get<1>(fd_c.dt_tilde_s), get<1>(fd_3.dt_tilde_s),
                         "dt tilde S", true);
  }
  {
    INFO("z component");
    if constexpr (Spherical) {
      {
        INFO("dt tilde B");
        CHECK_ITERABLE_APPROX(get<2>(fd_c.dt_tilde_b), get<2>(fd_3.dt_tilde_b));
      }
      {
        INFO("dt tilde S");
        CHECK_ITERABLE_APPROX(get<2>(fd_c.dt_tilde_s), get<2>(fd_3.dt_tilde_s));
      }
    } else {
      check_without_border(get<2>(fd_c.dt_tilde_b), get<2>(fd_3.dt_tilde_b),
                           "dt tilde B");
      check_without_border(get<2>(fd_c.dt_tilde_s), get<2>(fd_3.dt_tilde_s),
                           "dt tilde S");
    }
  }
}

// Evaluates the RHS for a full 3D evolution and a cartoon evolution and
// checks they are the same (cartoon sources are not needed, as the cartoon
// divergence function properly handles the flux terms)
template <bool Spherical>
void test_cartoon_dg_time_derivative() {
  CAPTURE(Spherical);
  const double time = 0.0;
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      fot{};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3d =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto block_map =
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
          Affine3d{Affine{-1., 1., 1.5, 3.0}, Affine{-1., 1., -0.2, 0.2},
                   Affine{-1., 1., -0.2, 0.2}});
  const Block<3> block{block_map.get_clone(), 0, {}};
  const ElementId<3> element_id{0};
  const ElementMap<3, Frame::Grid> logical_to_grid_map{element_id, block};
  const auto grid_to_inertial_map_ptr =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});
  const auto& grid_to_inertial_map = *grid_to_inertial_map_ptr;

  const size_t Nx_3 = 11;
  const size_t Ny_3 = Nx_3;
  const size_t Nz_3 = Nx_3;
  const size_t Nx_c = Nx_3;
  const size_t Ny_c = Spherical ? 1 : Ny_3;
  const size_t Nz_c = 1;
  const Index<3> extents_3{Nx_3, Ny_3, Nz_3};
  const Index<3> extents_c{Nx_c, Ny_c, Nz_c};

  constexpr Spectral::Quadrature cartoon_q =
      Spherical ? Spectral::Quadrature::SphericalSymmetry
                : Spectral::Quadrature::AxialSymmetry;
  Mesh<3> cartoon_mesh{};
  Mesh<3> plain_mesh{};
  cartoon_mesh = Mesh<3>{
      extents_c.indices(),
      {{Spectral::Basis::Legendre,
        Spherical ? Spectral::Basis::Cartoon : Spectral::Basis::Legendre,
        Spectral::Basis::Cartoon}},
      {{Spectral::Quadrature::GaussLobatto,
        Spherical ? cartoon_q : Spectral::Quadrature::GaussLobatto,
        cartoon_q}}};
  plain_mesh = Mesh<3>{extents_3.indices(), Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};

  const auto logical_coords_c = logical_coordinates(cartoon_mesh);
  const auto inertial_coords_c =
      grid_to_inertial_map(logical_to_grid_map(logical_coords_c), time, fot);
  const size_t num_pts_c = extents_c.product();
  const auto logical_coords_3 = logical_coordinates(plain_mesh);
  const auto inertial_coords_3 =
      grid_to_inertial_map(logical_to_grid_map(logical_coords_3), time, fot);

  const auto extract_slice = [num_pts_c](const DataVector& v3) -> DataVector {
    const size_t k_mid = Nz_3 / 2;
    const size_t j_mid = Ny_3 / 2;
    DataVector slice(num_pts_c);
    if constexpr (Spherical) {
      for (size_t ii = 0; ii < Nx_c; ++ii) {
        slice[ii] = v3[ii + Nx_3 * (j_mid + Ny_3 * k_mid)];
      }
    } else {
      for (size_t jj = 0; jj < Ny_c; ++jj) {
        for (size_t ii = 0; ii < Nx_c; ++ii) {
          slice[ii + Nx_c * jj] = v3[ii + Nx_3 * (jj + Ny_3 * k_mid)];
        }
      }
    }
    return slice;
  };

  const grmhd::Solutions::BondiMichel soln{1.0, 5.0, 0.05, 1.4, 2.0};
  constexpr double constraint_damping_parameter = 0.0;

  // Tags requested from the BondiMichel solution.
  using SolnTags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>>;

  // lambda to compute du/dt of conserved variables and put in a HydroResults
  // struct for comparison between full 3D and cartoon timestep
  const auto dg_compute = [&soln, time, &element_id, &block, &extract_slice](
                              const Mesh<3>& mesh,
                              const tnsr::I<DataVector, 3,
                                            Frame::ElementLogical>&
                                  logical_coords,
                              const tnsr::I<DataVector, 3>& inertial_coords) {
    auto vars = soln.variables(inertial_coords, time, SolnTags{});

    // Apply a Gaussian perturbation to make du/dt nontrivial
    constexpr double eps_perturb = 0.1;
    constexpr double r0_perturb = 2.25;
    constexpr double sigma_perturb = 0.5;
    constexpr double gamma_eos = 1.4;

    const double K_poly =
        get(get<hydro::Tags::Pressure<DataVector>>(vars))[0] /
        std::pow(get(get<hydro::Tags::RestMassDensity<DataVector>>(vars))[0],
                 gamma_eos);

    const size_t num_pts = mesh.number_of_grid_points();
    DataVector f(num_pts);
    const DataVector& xn = get<0>(inertial_coords);
    const DataVector& yn = get<1>(inertial_coords);
    const DataVector& zn = get<2>(inertial_coords);
    const DataVector r = [&xn, &yn, &zn]() -> DataVector {
      if constexpr (Spherical) {
        return sqrt(xn * xn + yn * yn + zn * zn);
      } else {
        (void)yn;
        return sqrt(xn * xn + zn * zn);
      }
    }();
    const DataVector dr = r - r0_perturb;
    f = exp(-dr * dr / (2 * sigma_perturb * sigma_perturb));

    // Perturbed 3D primitives
    Scalar<DataVector> rho(num_pts, 0.0);
    get(rho) = get(get<hydro::Tags::RestMassDensity<DataVector>>(vars)) *
               (1.0 + eps_perturb * f);
    Scalar<DataVector> p(num_pts, 0.0);
    Scalar<DataVector> eps_int(num_pts, 0.0);
    for (size_t n = 0; n < num_pts; ++n) {
      get(p)[n] = K_poly * std::pow(get(rho)[n], gamma_eos);
    }
    get(eps_int) = get(p) / (get(rho) * (gamma_eos - 1.0));
    Scalar<DataVector> phi(num_pts, 0.0);
    get(phi) =
        get(get<hydro::Tags::DivergenceCleaningField<DataVector>>(vars)) +
        eps_perturb * f;

    Scalar<DataVector> tilde_d(num_pts);
    Scalar<DataVector> tilde_ye(num_pts);
    Scalar<DataVector> tilde_tau(num_pts);
    tnsr::i<DataVector, 3, Frame::Inertial> tilde_s(num_pts);
    tnsr::I<DataVector, 3, Frame::Inertial> tilde_b(num_pts);
    Scalar<DataVector> tilde_phi(num_pts);
    grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
        make_not_null(&tilde_d), make_not_null(&tilde_ye),
        make_not_null(&tilde_tau), make_not_null(&tilde_s),
        make_not_null(&tilde_b), make_not_null(&tilde_phi), rho,
        get<hydro::Tags::ElectronFraction<DataVector>>(vars), eps_int, p,
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(vars),
        get<hydro::Tags::LorentzFactor<DataVector>>(vars),
        get<hydro::Tags::MagneticField<DataVector, 3>>(vars),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(vars), phi);

    Variables<
        typename grmhd::ValenciaDivClean::TimeDerivativeTerms::temporary_tags>
        temp_vars(num_pts);

    // Output variables for time derivatives
    Scalar<DataVector> dt_tilde_d(num_pts, 0.0);
    Scalar<DataVector> dt_tilde_ye(num_pts, 0.0);
    Scalar<DataVector> dt_tilde_tau(num_pts, 0.0);
    tnsr::i<DataVector, 3, Frame::Inertial> dt_tilde_s(num_pts, 0.0);
    tnsr::I<DataVector, 3, Frame::Inertial> dt_tilde_b(num_pts, 0.0);
    Scalar<DataVector> dt_tilde_phi(num_pts, 0.0);

    // Flux variables
    tnsr::I<DataVector, 3, Frame::Inertial> tilde_d_flux(num_pts, 0.0);
    tnsr::I<DataVector, 3, Frame::Inertial> tilde_ye_flux(num_pts, 0.0);
    tnsr::I<DataVector, 3, Frame::Inertial> tilde_tau_flux(num_pts, 0.0);
    tnsr::Ij<DataVector, 3, Frame::Inertial> tilde_s_flux(num_pts, 0.0);
    tnsr::IJ<DataVector, 3, Frame::Inertial> tilde_b_flux(num_pts, 0.0);
    tnsr::I<DataVector, 3, Frame::Inertial> tilde_phi_flux(num_pts, 0.0);

    grmhd::ValenciaDivClean::TimeDerivativeTerms::apply(
        make_not_null(&dt_tilde_d), make_not_null(&dt_tilde_ye),
        make_not_null(&dt_tilde_tau), make_not_null(&dt_tilde_s),
        make_not_null(&dt_tilde_b), make_not_null(&dt_tilde_phi),

        make_not_null(&tilde_d_flux), make_not_null(&tilde_ye_flux),
        make_not_null(&tilde_tau_flux), make_not_null(&tilde_s_flux),
        make_not_null(&tilde_b_flux), make_not_null(&tilde_phi_flux),

        // Temporary variables
        make_not_null(&get<hydro::Tags::SpatialVelocityOneForm<
                          DataVector, 3, Frame::Inertial>>(temp_vars)),
        make_not_null(&get<hydro::Tags::MagneticFieldOneForm<
                          DataVector, 3, Frame::Inertial>>(temp_vars)),
        make_not_null(
            &get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
                temp_vars)),
        make_not_null(
            &get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_vars)),
        make_not_null(&get<grmhd::ValenciaDivClean::TimeDerivativeTerms::
                               OneOverLorentzFactorSquared>(temp_vars)),
        make_not_null(
            &get<grmhd::ValenciaDivClean::TimeDerivativeTerms::PressureStar>(
                temp_vars)),
        make_not_null(
            &get<grmhd::ValenciaDivClean::TimeDerivativeTerms::
                     PressureStarLapseSqrtDetSpatialMetric>(temp_vars)),
        make_not_null(
            &get<
                hydro::Tags::TransportVelocity<DataVector, 3, Frame::Inertial>>(
                temp_vars)),
        make_not_null(
            &get<
                grmhd::ValenciaDivClean::TimeDerivativeTerms::LapseTimesbOverW>(
                temp_vars)),

        // More temporary variables for sources
        make_not_null(
            &get<grmhd::ValenciaDivClean::TimeDerivativeTerms::TildeSUp>(
                temp_vars)),
        make_not_null(
            &get<
                grmhd::ValenciaDivClean::TimeDerivativeTerms::DensitizedStress>(
                temp_vars)),
        make_not_null(
            &get<gr::Tags::SpatialChristoffelFirstKind<DataVector, 3>>(
                temp_vars)),
        make_not_null(
            &get<gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>>(
                temp_vars)),
        make_not_null(
            &get<gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, 3>>(
                temp_vars)),
        make_not_null(
            &get<grmhd::ValenciaDivClean::TimeDerivativeTerms::
                     EnthalpyTimesDensityWSquaredPlusBSquared>(temp_vars)),

        // Temp spacetime vars
        make_not_null(&get<gr::Tags::Lapse<DataVector>>(temp_vars)),
        make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(temp_vars)),
        make_not_null(
            &get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(temp_vars)),

        // Input variables
        tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,
        get<gr::Tags::Lapse<DataVector>>(vars),
        get<gr::Tags::Shift<DataVector, 3>>(vars),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(vars),
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(vars),
        get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                          Frame::Inertial>>(vars),
        get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                          Frame::Inertial>>(vars),
        get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                          tmpl::size_t<3>, Frame::Inertial>>(vars),
        p, get<hydro::Tags::SpatialVelocity<DataVector, 3>>(vars),
        get<hydro::Tags::LorentzFactor<DataVector>>(vars),
        get<hydro::Tags::MagneticField<DataVector, 3>>(vars), rho,
        get<hydro::Tags::ElectronFraction<DataVector>>(vars), eps_int,
        get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(vars),
        constraint_damping_parameter);

    // Subtract flux divergence to get full RHS = sources - div(F)
    {
      using FD = ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD,
                              tmpl::size_t<3>, Frame::Inertial>;
      using FYe = ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeYe,
                               tmpl::size_t<3>, Frame::Inertial>;
      using FTau = ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau,
                                tmpl::size_t<3>, Frame::Inertial>;
      using FS = ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeS<>,
                              tmpl::size_t<3>, Frame::Inertial>;
      using FB = ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeB<>,
                              tmpl::size_t<3>, Frame::Inertial>;
      using FPhi = ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi,
                                tmpl::size_t<3>, Frame::Inertial>;
      using FluxTags = tmpl::list<FD, FYe, FTau, FS, FB, FPhi>;
      Variables<FluxTags> volume_fluxes(num_pts);
      get<FD>(volume_fluxes) = tilde_d_flux;
      get<FYe>(volume_fluxes) = tilde_ye_flux;
      get<FTau>(volume_fluxes) = tilde_tau_flux;
      get<FS>(volume_fluxes) = tilde_s_flux;
      get<FB>(volume_fluxes) = tilde_b_flux;
      get<FPhi>(volume_fluxes) = tilde_phi_flux;

      // Need ElementLogical-to-Inertial inv_jac; since grid_to_inertial is
      // Identity, construct from block directly in Frame::Inertial
      const ElementMap<3, Frame::Inertial> elem_map_inertial{element_id, block};
      const auto inv_jac = elem_map_inertial.inv_jacobian(logical_coords);
      Variables<db::wrap_tags_in<Tags::div, FluxTags>> div_fluxes(num_pts);
      divergence(make_not_null(&div_fluxes), volume_fluxes, mesh, inv_jac,
                 inertial_coords);

      get(dt_tilde_d) -= get(get<::Tags::div<FD>>(div_fluxes));
      get(dt_tilde_ye) -= get(get<::Tags::div<FYe>>(div_fluxes));
      get(dt_tilde_tau) -= get(get<::Tags::div<FTau>>(div_fluxes));
      for (size_t i = 0; i < 3; ++i) {
        dt_tilde_s.get(i) -= get<::Tags::div<FS>>(div_fluxes).get(i);
        dt_tilde_b.get(i) -= get<::Tags::div<FB>>(div_fluxes).get(i);
      }
      get(dt_tilde_phi) -= get(get<::Tags::div<FPhi>>(div_fluxes));
    }

    const size_t num_points =
        Spherical ? mesh.slice_through(0).number_of_grid_points()
                  : mesh.slice_away(2).number_of_grid_points();
    HydroResults result{
        Scalar<DataVector>(num_points, 0.0),
        Scalar<DataVector>(num_points, 0.0),
        Scalar<DataVector>(num_points, 0.0),
        tnsr::i<DataVector, 3>(num_points, 0.0),
        tnsr::I<DataVector, 3>(num_points, 0.0),
        Scalar<DataVector>(num_points, 0.0),
    };
    if (mesh.basis(2) != Spectral::Basis::Cartoon) {
      get(result.dt_tilde_d) = extract_slice(get(dt_tilde_d));
      get(result.dt_tilde_ye) = extract_slice(get(dt_tilde_ye));
      get(result.dt_tilde_tau) = extract_slice(get(dt_tilde_tau));
      get(result.dt_tilde_phi) = extract_slice(get(dt_tilde_phi));
      for (size_t i = 0; i < 3; ++i) {
        result.dt_tilde_s.get(i) = extract_slice(dt_tilde_s.get(i));
        result.dt_tilde_b.get(i) = extract_slice(dt_tilde_b.get(i));
      }
    } else {
      get(result.dt_tilde_d) = get(dt_tilde_d);
      get(result.dt_tilde_ye) = get(dt_tilde_ye);
      get(result.dt_tilde_tau) = get(dt_tilde_tau);
      get(result.dt_tilde_phi) = get(dt_tilde_phi);
      for (size_t i = 0; i < 3; ++i) {
        result.dt_tilde_s.get(i) = dt_tilde_s.get(i);
        result.dt_tilde_b.get(i) = dt_tilde_b.get(i);
      }
    }
    return result;
  };
  const auto dg_c =
      dg_compute(cartoon_mesh, logical_coords_c, inertial_coords_c);
  const auto dg_3 = dg_compute(plain_mesh, logical_coords_3, inertial_coords_3);

  const auto local_approx = Approx::custom().epsilon(1e-5).scale(1e-11);
  constexpr size_t comp_dim = Spherical ? 1 : 2;
  CHECK_ITERABLE_CUSTOM_APPROX(dg_c.dt_tilde_d, dg_3.dt_tilde_d, local_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dg_c.dt_tilde_ye, dg_3.dt_tilde_ye,
                               local_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dg_c.dt_tilde_tau, dg_3.dt_tilde_tau,
                               local_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dg_c.dt_tilde_phi, dg_3.dt_tilde_phi,
                               local_approx);
  for (size_t i = 0; i < 3; ++i) {
    CAPTURE(i);
    CHECK_ITERABLE_CUSTOM_APPROX(dg_c.dt_tilde_b.get(i), dg_3.dt_tilde_b.get(i),
                                 i < comp_dim ? local_approx : approx);
    CHECK_ITERABLE_CUSTOM_APPROX(dg_c.dt_tilde_s.get(i), dg_3.dt_tilde_s.get(i),
                                 i < comp_dim ? local_approx : approx);
  }
}

#ifdef SPECTRE_DEBUG
void test_cartoon_sources_x_zero_assert() {
  const size_t num_pts = 3;
  tnsr::i<DataVector, 3, Frame::Inertial> source_tilde_s(num_pts, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> source_tilde_b(num_pts, 0.0);

  const Scalar<DataVector> pressure_star(num_pts, 1.0);
  const tnsr::i<DataVector, 3, Frame::Inertial> magnetic_field_one_form(num_pts,
                                                                        0.0);
  const Scalar<DataVector> magnetic_field_dot_spatial_velocity(num_pts, 0.0);
  const tnsr::i<DataVector, 3, Frame::Inertial> tilde_s(num_pts, 0.0);
  const tnsr::I<DataVector, 3, Frame::Inertial> tilde_b(num_pts, 0.0);
  const Scalar<DataVector> tilde_phi(num_pts, 0.0);
  const tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity(num_pts, 0.0);
  const Scalar<DataVector> lorentz_factor(num_pts, 1.0);
  const Scalar<DataVector> lapse(num_pts, 1.0);
  const tnsr::I<DataVector, 3, Frame::Inertial> shift(num_pts, 0.0);

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric(num_pts, 0.0);
  get<0, 0>(spatial_metric) = 1.0;
  get<1, 1>(spatial_metric) = 1.0;
  get<2, 2>(spatial_metric) = 1.0;

  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric(num_pts, 0.0);
  get<0, 0>(inv_spatial_metric) = 1.0;
  get<1, 1>(inv_spatial_metric) = 1.0;
  get<2, 2>(inv_spatial_metric) = 1.0;

  const Scalar<DataVector> sqrt_det_spatial_metric(num_pts, 1.0);
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords(num_pts, 0.0);

  // First x-coordinate is zero
  get<0>(inertial_coords) = DataVector{0.0, 1.0, 2.0};
  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::detail::cartoon_sources_impl(
          make_not_null(&source_tilde_s), make_not_null(&source_tilde_b),
          pressure_star, magnetic_field_one_form,
          magnetic_field_dot_spatial_velocity, tilde_s, tilde_b, tilde_phi,
          spatial_velocity, lorentz_factor, lapse, shift, spatial_metric,
          inv_spatial_metric, sqrt_det_spatial_metric, inertial_coords,
          Spectral::Quadrature::SphericalSymmetry),
      Catch::Matchers::ContainsSubstring(
          "Cannot compute the Cartoon source terms with x=0"));

  // An in-between x-coordinate is zero
  get<0>(inertial_coords) = DataVector{1.0, 0.0, 2.0};
  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::detail::cartoon_sources_impl(
          make_not_null(&source_tilde_s), make_not_null(&source_tilde_b),
          pressure_star, magnetic_field_one_form,
          magnetic_field_dot_spatial_velocity, tilde_s, tilde_b, tilde_phi,
          spatial_velocity, lorentz_factor, lapse, shift, spatial_metric,
          inv_spatial_metric, sqrt_det_spatial_metric, inertial_coords,
          Spectral::Quadrature::SphericalSymmetry),
      Catch::Matchers::ContainsSubstring(
          "Cannot compute the Cartoon source terms with x=0"));
}
#endif  // SPECTRE_DEBUG
}  // namespace

// [[TimeOut, 20]]
SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Sources", "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  pypp::check_with_random_values<1>(&apply_standard_sources, "Sources",
                                    {"source_tilde_tau", "source_tilde_s",
                                     "source_tilde_b", "source_tilde_phi"},
                                    {{{0.0, 1.0}}}, DataVector{5});

  // Test DG time derivative
  test_cartoon_dg_time_derivative<true>();
  test_cartoon_dg_time_derivative<false>();

  // Test FD time derivative
  test_cartoon_fd_time_derivative<true>();
  test_cartoon_fd_time_derivative<false>();

#ifdef SPECTRE_DEBUG
  test_cartoon_sources_x_zero_assert();
#endif  // SPECTRE_DEBUG
}
