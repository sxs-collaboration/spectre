// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/ElementMap.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/Factory.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            grmhd::GhValenciaDivClean::BoundaryConditions::BoundaryCondition,
            tmpl::list<grmhd::GhValenciaDivClean::BoundaryConditions::
                           DirichletAnalytic>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   ghmhd::GhValenciaDivClean::InitialData::initial_data_list>,
        tmpl::pair<
            grmhd::AnalyticData::InitialMagneticFields::InitialMagneticField,
            tmpl::list<grmhd::AnalyticData::InitialMagneticFields::Poloidal>>>;
  };
};

template <typename T, typename U>
void test_dg(const gsl::not_null<std::mt19937*> generator,
             const U& boundary_condition, const T& analytic_solution_or_data) {
  const double time = 1.3;
  const size_t num_points = 5;

  std::uniform_real_distribution<> dist(0.1, 1.0);

  const auto interior_gamma1 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist), num_points);
  const auto interior_gamma2 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist), num_points);

  const auto coords =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist), num_points);

  using Vars = Variables<tmpl::pop_back<
      grmhd::GhValenciaDivClean::fd::tags_list_for_reconstruct_fd_neighbor>>;
  using tags_not_set_by_boundary_condition =
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>>;
  Vars vars{num_points};
  const auto expected_vars = [&analytic_solution_or_data, &coords,
                              &interior_gamma1, &interior_gamma2, time]() {
    Vars expected{num_points};
    auto& [spacetime_metric, pi, phi, tilde_d, tilde_ye, tilde_tau, tilde_s,
           tilde_b, tilde_phi,

           rest_mass_density, electron_fraction, specific_internal_energy,
           spatial_velocity, magnetic_field, divergence_cleaning_field,
           lorentz_factor, pressure, temperature,
           lorentz_factor_times_spatial_velocity,

           tilde_d_flux, tilde_ye_flux, tilde_tau_flux, tilde_s_flux,
           tilde_b_flux, tilde_phi_flux, gamma1, gamma2, lapse, shift,

           spatial_velocity_one_form, spatial_metric, sqrt_det_spatial_metric,

           inverse_spatial_metric] = expected;

    gamma1 = interior_gamma1;
    gamma2 = interior_gamma2;

    using tags =
        tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                   hydro::Tags::ElectronFraction<DataVector>,
                   hydro::Tags::Temperature<DataVector>,
                   hydro::Tags::SpecificInternalEnergy<DataVector>,
                   hydro::Tags::SpecificEnthalpy<DataVector>,
                   hydro::Tags::Pressure<DataVector>,
                   hydro::Tags::SpatialVelocity<DataVector, 3>,
                   hydro::Tags::LorentzFactor<DataVector>,
                   hydro::Tags::MagneticField<DataVector, 3>,
                   hydro::Tags::DivergenceCleaningField<DataVector>,
                   gr::Tags::SpatialMetric<DataVector, 3>,
                   gr::Tags::InverseSpatialMetric<DataVector, 3>,
                   gr::Tags::SqrtDetSpatialMetric<DataVector>,
                   gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                   gr::Tags::SpacetimeMetric<DataVector, 3>,
                   ::gh::Tags::Pi<DataVector, 3>,
                   ::gh::Tags::Phi<DataVector, 3>>;

    tuples::tagged_tuple_from_typelist<tags> analytic_vars{};

    if constexpr (::is_analytic_solution_v<T>) {
      analytic_vars = analytic_solution_or_data.variables(coords, time, tags{});
    } else {
      (void)time;
      analytic_vars = analytic_solution_or_data.variables(coords, tags{});
    }
    expected.assign_subset(analytic_vars);
    tenex::evaluate<ti::i>(
        make_not_null(&spatial_velocity_one_form),
        spatial_velocity(ti::J) * spatial_metric(ti::i, ti::j));

    grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
        make_not_null(&tilde_d), make_not_null(&tilde_ye),
        make_not_null(&tilde_tau), make_not_null(&tilde_s),
        make_not_null(&tilde_b), make_not_null(&tilde_phi),
        get<hydro::Tags::RestMassDensity<DataVector>>(analytic_vars),
        get<hydro::Tags::ElectronFraction<DataVector>>(analytic_vars),
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(analytic_vars),
        get<hydro::Tags::Pressure<DataVector>>(analytic_vars),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(analytic_vars),
        get<hydro::Tags::LorentzFactor<DataVector>>(analytic_vars),
        get<hydro::Tags::MagneticField<DataVector, 3>>(analytic_vars),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(analytic_vars),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(analytic_vars),
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(analytic_vars));

    grmhd::ValenciaDivClean::ComputeFluxes::apply(
        make_not_null(&tilde_d_flux), make_not_null(&tilde_ye_flux),
        make_not_null(&tilde_tau_flux), make_not_null(&tilde_s_flux),
        make_not_null(&tilde_b_flux), make_not_null(&tilde_phi_flux), tilde_d,
        tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(analytic_vars),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(analytic_vars),
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(analytic_vars),
        get<hydro::Tags::Pressure<DataVector>>(analytic_vars),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(analytic_vars),
        get<hydro::Tags::LorentzFactor<DataVector>>(analytic_vars),
        get<hydro::Tags::MagneticField<DataVector, 3>>(analytic_vars));

    return expected;
  }();
  auto& [spacetime_metric, pi, phi, tilde_d, tilde_ye, tilde_tau, tilde_s,
         tilde_b, tilde_phi,

         rest_mass_density, electron_fraction, specific_internal_energy,
         spatial_velocity, magnetic_field, divergence_cleaning_field,
         lorentz_factor, pressure, temperature,
         lorentz_factor_times_spatial_velocity,

         tilde_d_flux, tilde_ye_flux, tilde_tau_flux, tilde_s_flux,
         tilde_b_flux, tilde_phi_flux, gamma1, gamma2, lapse, shift,

         spatial_velocity_one_form, spatial_metric, sqrt_det_spatial_metric,

         inverse_spatial_metric] = vars;

  const auto result = boundary_condition.dg_ghost(
      make_not_null(&spacetime_metric), make_not_null(&pi), make_not_null(&phi),
      make_not_null(&tilde_d), make_not_null(&tilde_ye),
      make_not_null(&tilde_tau), make_not_null(&tilde_s),
      make_not_null(&tilde_b), make_not_null(&tilde_phi),
      make_not_null(&tilde_d_flux), make_not_null(&tilde_ye_flux),
      make_not_null(&tilde_tau_flux), make_not_null(&tilde_s_flux),
      make_not_null(&tilde_b_flux), make_not_null(&tilde_phi_flux),
      make_not_null(&gamma1), make_not_null(&gamma2), make_not_null(&lapse),
      make_not_null(&shift), make_not_null(&spatial_velocity_one_form),
      make_not_null(&rest_mass_density), make_not_null(&electron_fraction),
      make_not_null(&temperature), make_not_null(&spatial_velocity),
      make_not_null(&inverse_spatial_metric), {}, {}, {}, coords,
      interior_gamma1, interior_gamma2, time);
  CHECK(not result.has_value());

  tmpl::for_each<typename Vars::tags_list>([&expected_vars, &vars](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    if constexpr (not tmpl::list_contains_v<tags_not_set_by_boundary_condition,
                                            tag>) {
      CAPTURE(db::tag_name<tag>());
      CHECK(get<tag>(vars) == get<tag>(expected_vars));
    }
  });
}

template <typename T, typename U>
void test_fd(const U& boundary_condition, const T& analytic_solution_or_data) {
  std::uniform_real_distribution<> dist(0.1, 1.0);

  const double time = 1.3;
  const Mesh<3> subcell_mesh{9, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};

  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  const std::array<double, 3> lower_bound{{0.78, 1.18, 1.28}};
  const std::array<double, 3> upper_bound{{0.82, 1.22, 1.32}};
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]},
                   Affine{-1., 1., lower_bound[2], upper_bound[2]}});
  const ElementId<3> element_id{0};
  const ElementMap logical_to_grid_map{
      element_id,
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
          Affine3D{Affine{-1., 1., 2.0 * lower_bound[0], 2.0 * upper_bound[0]},
                   Affine{-1., 1., 2.0 * lower_bound[1], 2.0 * upper_bound[1]},
                   Affine{-1., 1., 2.0 * lower_bound[2], 2.0 * upper_bound[2]}})
          .get_clone()};
  const auto direction = Direction<3>::lower_xi();

  grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim reconstructor{};

  using Vars = Variables<grmhd::GhValenciaDivClean::Tags::
                             primitive_grmhd_and_spacetime_reconstruction_tags>;
  Vars vars{reconstructor.ghost_zone_size() *
            subcell_mesh.extents().slice_away(0).product()};
  const auto expected_vars = [&analytic_solution_or_data, &direction,
                              &functions_of_time, &grid_to_inertial_map,
                              &logical_to_grid_map, &reconstructor,
                              &subcell_mesh, time]() {
    const auto ghost_logical_coords =
        evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
            subcell_mesh, reconstructor.ghost_zone_size(), direction);

    const auto ghost_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(ghost_logical_coords), time, functions_of_time);

    using tags = tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                            hydro::Tags::ElectronFraction<DataVector>,
                            hydro::Tags::Temperature<DataVector>,
                            hydro::Tags::SpatialVelocity<DataVector, 3>,
                            hydro::Tags::LorentzFactor<DataVector>,
                            hydro::Tags::MagneticField<DataVector, 3>,
                            hydro::Tags::DivergenceCleaningField<DataVector>,
                            gr::Tags::SpacetimeMetric<DataVector, 3>,
                            ::gh::Tags::Pi<DataVector, 3>,
                            ::gh::Tags::Phi<DataVector, 3>>;

    tuples::tagged_tuple_from_typelist<tags> analytic_vars{};

    if constexpr (::is_analytic_solution_v<T>) {
      analytic_vars = analytic_solution_or_data.variables(ghost_inertial_coords,
                                                          time, tags{});
    } else {
      (void)time;
      analytic_vars =
          analytic_solution_or_data.variables(ghost_inertial_coords, tags{});
    }

    Vars expected{get<0>(ghost_inertial_coords).size()};

    get<gr::Tags::SpacetimeMetric<DataVector, 3>>(expected) =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(analytic_vars);
    get<::gh::Tags::Pi<DataVector, 3>>(expected) =
        get<::gh::Tags::Pi<DataVector, 3>>(analytic_vars);
    get<::gh::Tags::Phi<DataVector, 3>>(expected) =
        get<::gh::Tags::Phi<DataVector, 3>>(analytic_vars);
    get<hydro::Tags::RestMassDensity<DataVector>>(expected) =
        get<hydro::Tags::RestMassDensity<DataVector>>(analytic_vars);
    get<hydro::Tags::ElectronFraction<DataVector>>(expected) =
        get<hydro::Tags::ElectronFraction<DataVector>>(analytic_vars);
    get<hydro::Tags::Temperature<DataVector>>(expected) =
        get<hydro::Tags::Temperature<DataVector>>(analytic_vars);

    for (size_t i = 0; i < 3; ++i) {
      auto& lorentz_factor =
          get<hydro::Tags::LorentzFactor<DataVector>>(analytic_vars);
      auto& spatial_velocity =
          get<hydro::Tags::SpatialVelocity<DataVector, 3>>(analytic_vars);
      get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
          expected)
          .get(i) = get(lorentz_factor) * spatial_velocity.get(i);
    }

    get<hydro::Tags::MagneticField<DataVector, 3>>(expected) =
        get<hydro::Tags::MagneticField<DataVector, 3>>(analytic_vars);
    get<hydro::Tags::DivergenceCleaningField<DataVector>>(expected) =
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(analytic_vars);
    return expected;
  }();
  auto& [rest_mass_density, electron_fraction, temperature,
         lorentz_factor_times_spatial_velocity, magnetic_field,
         divergence_cleaning_field, spacetime_metric, pi, phi] = vars;

  boundary_condition.fd_ghost(
      make_not_null(&spacetime_metric), make_not_null(&pi), make_not_null(&phi),
      make_not_null(&rest_mass_density), make_not_null(&electron_fraction),
      make_not_null(&temperature),
      make_not_null(&lorentz_factor_times_spatial_velocity),
      make_not_null(&magnetic_field), make_not_null(&divergence_cleaning_field),
      direction, subcell_mesh, time, functions_of_time, logical_to_grid_map,
      grid_to_inertial_map, reconstructor);
  // failing line
  CHECK(vars == expected_vars);
}

SPECTRE_TEST_CASE(
    "Unit.GhValenciaDivClean.BoundaryConditions.DirichletAnalytic",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  register_factory_classes_with_charm<Metavariables>();
  EquationsOfState::register_derived_with_charm();
  {
    INFO("Test with analytic solution");
    const auto product_boundary_condition =
        TestHelpers::test_creation<
            std::unique_ptr<grmhd::GhValenciaDivClean::BoundaryConditions::
                                BoundaryCondition>,
            Metavariables>(
            "DirichletAnalytic:\n"
            "  AnalyticPrescription:\n"
            "      GeneralizedHarmonic(BondiMichel):\n"
            "        Mass: 1.0\n"
            "        SonicRadius: 4.0\n"
            "        SonicDensity: 0.1\n"
            "        PolytropicExponent: 2.0\n"
            "        MagFieldStrength: 0.01\n")
            ->get_clone();

    const gh::Solutions::WrappedGr<grmhd::Solutions::BondiMichel>
        analytic_solution_or_data{1.0, 4.0, 0.1, 2.0, 0.01};
    const auto serialized_and_deserialized_condition =
        serialize_and_deserialize(
            *dynamic_cast<grmhd::GhValenciaDivClean::BoundaryConditions::
                              DirichletAnalytic*>(
                product_boundary_condition.get()));

    test_dg(make_not_null(&gen), serialized_and_deserialized_condition,
            analytic_solution_or_data);
    test_fd(serialized_and_deserialized_condition, analytic_solution_or_data);
  }
  {
    INFO("Test with analytic data");
    const auto product_boundary_condition =
        TestHelpers::test_creation<
            std::unique_ptr<grmhd::GhValenciaDivClean::BoundaryConditions::
                                BoundaryCondition>,
            Metavariables>(
            "DirichletAnalytic:\n"
            "  AnalyticPrescription:\n"
            "      GeneralizedHarmonic(MagnetizedTovStar):\n"
            "        CentralDensity: 1.28e-3\n"
            "        EquationOfState:\n"
            "          PolytropicFluid:\n"
            "            PolytropicConstant: 100.0\n"
            "            PolytropicExponent: 2.0\n"
            "        Coordinates: Schwarzschild\n"
            "        MagneticFields: \n"
            "          - Poloidal:\n"
            "              PressureExponent: 2\n"
            "              CutoffPressure: 0.04\n"
            "              VectorPotentialAmplitude: 2500\n"
            "              Center: [0.0, 0.0, 0.0]\n"
            "              MaxDistanceFromCenter: 100.0\n")
            ->get_clone();

    const gh::Solutions::WrappedGr<grmhd::AnalyticData::MagnetizedTovStar>
        analytic_solution_or_data{
            1.28e-3,
            std::make_unique<EquationsOfState::PolytropicFluid<true>>(100.0,
                                                                      2.0),
            RelativisticEuler::Solutions::TovCoordinates::Schwarzschild,
            make_vector<
                std::unique_ptr<grmhd::AnalyticData::InitialMagneticFields::
                                    InitialMagneticField>>(
                std::make_unique<
                    grmhd::AnalyticData::InitialMagneticFields::Poloidal>(
                    2, 0.04, 2500.0, std::array{0.0, 0.0, 0.0}, 100.0))};
    const auto serialized_and_deserialized_condition =
        serialize_and_deserialize(
            *dynamic_cast<grmhd::GhValenciaDivClean::BoundaryConditions::
                              DirichletAnalytic*>(
                product_boundary_condition.get()));

    test_dg(make_not_null(&gen), serialized_and_deserialized_condition,
            analytic_solution_or_data);
    test_fd(serialized_and_deserialized_condition, analytic_solution_or_data);
  }
}
}  // namespace
