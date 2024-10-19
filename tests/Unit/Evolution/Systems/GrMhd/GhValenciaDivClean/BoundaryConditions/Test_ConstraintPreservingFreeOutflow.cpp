// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/ConstraintPreservingFreeOutflow.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/HydroFreeOutflow.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        grmhd::GhValenciaDivClean::BoundaryConditions::BoundaryCondition,
        tmpl::list<grmhd::GhValenciaDivClean::BoundaryConditions::
                       ConstraintPreservingFreeOutflow>>>;
  };
};

template <typename U>
void test_dg(const gsl::not_null<std::mt19937*> generator,
             const U& boundary_condition) {
  const double time = 1.3;
  const size_t num_points = 5;

  std::uniform_real_distribution<> dist(0.1, 1.0);

  const gh::Solutions::WrappedGr<grmhd::Solutions::BondiMichel>
      analytic_solution{1.0, 4.0, 0.1, 2.0, 0.01};

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
  using PrimVars =
      Variables<tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                           hydro::Tags::ElectronFraction<DataVector>,
                           hydro::Tags::SpecificInternalEnergy<DataVector>,
                           hydro::Tags::SpecificEnthalpy<DataVector>,
                           hydro::Tags::Pressure<DataVector>,
                           hydro::Tags::Temperature<DataVector>,
                           hydro::Tags::SpatialVelocity<DataVector, 3>,
                           hydro::Tags::LorentzFactor<DataVector>,
                           hydro::Tags::MagneticField<DataVector, 3>>>;
  Vars vars{num_points};
  Vars expected_vars;
  PrimVars prim_vars;

  std::tie(expected_vars, prim_vars) = [&analytic_solution, &coords,
                                        &interior_gamma1, &interior_gamma2,
                                        time]() {
    Vars expected{num_points};
    auto& [spacetime_metric, pi, phi, tilde_d, tilde_ye, tilde_tau, tilde_s,
           tilde_b, tilde_phi,

           rest_mass_density, electron_fraction, specific_internal_energy,
           spatial_velocity, magnetic_field, divergence_cleaning_field,
           lorentz_factor, pressure, temperature,
           lorentz_factor_times_spatial_velocity,

           tilde_d_flux, tilde_ye_flux, tilde_tau_flux, tilde_s_flux,
           tilde_b_flux, tilde_phi_flux,

           gamma1, gamma2, lapse, shift,

           spatial_velocity_one_form, spatial_metric, sqrt_det_spatial_metric,
           inverse_spatial_metric] = expected;

    gamma1 = interior_gamma1;
    gamma2 = interior_gamma2;

    PrimVars local_prim_vars{num_points};

    using tags = tmpl::list<
        hydro::Tags::RestMassDensity<DataVector>,
        hydro::Tags::ElectronFraction<DataVector>,
        hydro::Tags::SpecificInternalEnergy<DataVector>,
        hydro::Tags::SpecificEnthalpy<DataVector>,
        hydro::Tags::Pressure<DataVector>, hydro::Tags::Temperature<DataVector>,
        hydro::Tags::SpatialVelocity<DataVector, 3>,
        hydro::Tags::LorentzFactor<DataVector>,
        hydro::Tags::MagneticField<DataVector, 3>,
        hydro::Tags::DivergenceCleaningField<DataVector>,
        gr::Tags::SpatialMetric<DataVector, 3>,
        gr::Tags::InverseSpatialMetric<DataVector, 3>,
        gr::Tags::SqrtDetSpatialMetric<DataVector>, gr::Tags::Lapse<DataVector>,
        gr::Tags::Shift<DataVector, 3>,
        gr::Tags::SpacetimeMetric<DataVector, 3>, ::gh::Tags::Pi<DataVector, 3>,
        ::gh::Tags::Phi<DataVector, 3>>;

    tuples::tagged_tuple_from_typelist<tags> analytic_vars{};

    analytic_vars = analytic_solution.variables(coords, time, tags{});
    local_prim_vars.assign_subset(analytic_vars);
    spacetime_metric =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(analytic_vars);
    pi = get<::gh::Tags::Pi<DataVector, 3>>(analytic_vars);
    phi = get<::gh::Tags::Phi<DataVector, 3>>(analytic_vars);
    lapse = get<gr::Tags::Lapse<DataVector>>(analytic_vars);
    shift = get<gr::Tags::Shift<DataVector, 3>>(analytic_vars);
    inverse_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(analytic_vars);

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

    return std::tuple(expected, local_prim_vars);
  }();

  // Pick random direction normal covector, then normalize and compute normal
  // vector.
  tnsr::i<DataVector, 3> normal_covector{num_points};
  get<0>(normal_covector) = 0.5;
  get<1>(normal_covector) = 0.0;
  get<2>(normal_covector) = 0.5;
  const auto magnitude_normal = magnitude(
      normal_covector,
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(expected_vars));
  for (size_t i = 0; i < 3; ++i) {
    normal_covector.get(i) /= get(magnitude_normal);
  }
  const auto normal_vector =
      tenex::evaluate<ti::I>(normal_covector(ti::j) *
                             get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                                 expected_vars)(ti::I, ti::J));

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

  CHECK(
      not boundary_condition
              .dg_ghost(
                  make_not_null(&spacetime_metric), make_not_null(&pi),
                  make_not_null(&phi), make_not_null(&tilde_d),
                  make_not_null(&tilde_ye), make_not_null(&tilde_tau),
                  make_not_null(&tilde_s), make_not_null(&tilde_b),
                  make_not_null(&tilde_phi), make_not_null(&tilde_d_flux),
                  make_not_null(&tilde_ye_flux), make_not_null(&tilde_tau_flux),
                  make_not_null(&tilde_s_flux), make_not_null(&tilde_b_flux),
                  make_not_null(&tilde_phi_flux), make_not_null(&gamma1),
                  make_not_null(&gamma2), make_not_null(&lapse),
                  make_not_null(&shift),
                  make_not_null(&spatial_velocity_one_form),
                  make_not_null(&rest_mass_density),
                  make_not_null(&electron_fraction),
                  make_not_null(&temperature), make_not_null(&spatial_velocity),
                  make_not_null(&inverse_spatial_metric), {}, normal_covector,
                  normal_vector,

                  get<gr::Tags::SpacetimeMetric<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Pi<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Phi<DataVector, 3>>(expected_vars),

                  get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars),
                  get<hydro::Tags::ElectronFraction<DataVector>>(prim_vars),
                  get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
                      prim_vars),
                  get<hydro::Tags::SpatialVelocity<DataVector, 3>>(prim_vars),
                  get<hydro::Tags::MagneticField<DataVector, 3>>(prim_vars),
                  get<hydro::Tags::LorentzFactor<DataVector>>(prim_vars),
                  get<hydro::Tags::Pressure<DataVector>>(prim_vars),
                  get<hydro::Tags::Temperature<DataVector>>(prim_vars),

                  coords, interior_gamma1, interior_gamma2,
                  get<gr::Tags::Lapse<DataVector>>(expected_vars),
                  get<gr::Tags::Shift<DataVector, 3>>(expected_vars),
                  get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                      expected_vars),
                  {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
              .has_value());

  const auto inverse_spacetime_metric =
      determinant_and_inverse(spacetime_metric).second;
  const auto spacetime_normal_vector = gr::spacetime_normal_vector(
      get<gr::Tags::Lapse<DataVector>>(expected_vars),
      get<gr::Tags::Shift<DataVector, 3>>(expected_vars));
  const auto gauge_source = make_with_random_values<tnsr::a<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);
  const auto spacetime_deriv_gauge_source =
      make_with_random_values<tnsr::ab<DataVector, 3>>(
          generator, make_not_null(&dist), num_points);
  const auto d_spacetime_metric =
      make_with_random_values<tnsr::iaa<DataVector, 3>>(
          generator, make_not_null(&dist), num_points);
  const auto d_pi = make_with_random_values<tnsr::iaa<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);
  const auto d_phi = make_with_random_values<tnsr::ijaa<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);
  const auto three_index_constraint =
      make_with_random_values<tnsr::iaa<DataVector, 3>>(
          generator, make_not_null(&dist), num_points);
  const auto logical_dt_spacetime_metric =
      make_with_random_values<tnsr::aa<DataVector, 3>>(
          generator, make_not_null(&dist), num_points);
  const auto logical_dt_pi = make_with_random_values<tnsr::aa<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);
  const auto logical_dt_phi = make_with_random_values<tnsr::iaa<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);

  using DtVars = Variables<db::wrap_tags_in<
      ::Tags::dt, grmhd::GhValenciaDivClean::System::variables_tag::tags_list>>;
  DtVars dt_vars{num_points};

  CHECK(
      not boundary_condition
              .dg_time_derivative(
                  make_not_null(
                      &get<
                          ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
                          dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>(dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<gh::Tags::Phi<DataVector, 3>>>(dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeD>>(
                          dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeYe>>(
                          dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeTau>>(
                          dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeS<
                          Frame::Inertial>>>(dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeB<
                          Frame::Inertial>>>(dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildePhi>>(
                          dt_vars)),

                  {}, normal_covector, normal_vector,

                  get<gr::Tags::SpacetimeMetric<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Pi<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Phi<DataVector, 3>>(expected_vars),

                  get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars),
                  get<hydro::Tags::ElectronFraction<DataVector>>(prim_vars),
                  get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
                      prim_vars),
                  get<hydro::Tags::SpatialVelocity<DataVector, 3>>(prim_vars),
                  get<hydro::Tags::MagneticField<DataVector, 3>>(prim_vars),
                  get<hydro::Tags::LorentzFactor<DataVector>>(prim_vars),
                  get<hydro::Tags::Pressure<DataVector>>(prim_vars),
                  get<hydro::Tags::Temperature<DataVector>>(prim_vars),

                  coords, interior_gamma1, interior_gamma2,
                  get<gr::Tags::Lapse<DataVector>>(expected_vars),
                  get<gr::Tags::Shift<DataVector, 3>>(expected_vars),
                  get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                      expected_vars),
                  inverse_spacetime_metric, spacetime_normal_vector,
                  three_index_constraint, gauge_source,
                  spacetime_deriv_gauge_source, logical_dt_spacetime_metric,
                  logical_dt_pi, logical_dt_phi, d_spacetime_metric, d_pi,
                  d_phi)
              .has_value());

  grmhd::ValenciaDivClean::BoundaryConditions::HydroFreeOutflow
      grmhd_free_outflow{};

  auto& [spacetime_metric_expected, pi_expected, phi_expected, tilde_d_expected,
         tilde_ye_expected, tilde_tau_expected, tilde_s_expected,
         tilde_b_expected, tilde_phi_expected,

         rest_mass_density_expected, electron_fraction_expected,
         specific_internal_energy_expected, spatial_velocity_expected,
         magnetic_field_expected, divergence_cleaning_field_expected,
         lorentz_factor_expected, pressure_expected, temperature_expected,
         lorentz_factor_times_spatial_velocity_expected,

         tilde_d_flux_expected, tilde_ye_flux_expected, tilde_tau_flux_expected,
         tilde_s_flux_expected, tilde_b_flux_expected, tilde_phi_flux_expected,
         gamma1_expected, gamma2_expected, lapse_expected, shift_expected,

         spatial_velocity_one_form_expected, spatial_metric_expected,
         sqrt_det_spatial_metric_expected, inverse_spatial_metric_expected] =
      expected_vars;

  CHECK(not grmhd_free_outflow
                .dg_ghost(
                    make_not_null(&tilde_d_expected),
                    make_not_null(&tilde_ye_expected),
                    make_not_null(&tilde_tau_expected),
                    make_not_null(&tilde_s_expected),
                    make_not_null(&tilde_b_expected),
                    make_not_null(&tilde_phi_expected),
                    make_not_null(&tilde_d_flux_expected),
                    make_not_null(&tilde_ye_flux_expected),
                    make_not_null(&tilde_tau_flux_expected),
                    make_not_null(&tilde_s_flux_expected),
                    make_not_null(&tilde_b_flux_expected),
                    make_not_null(&tilde_phi_flux_expected),
                    make_not_null(&lapse_expected),
                    make_not_null(&shift_expected),
                    make_not_null(&spatial_velocity_one_form_expected),
                    make_not_null(&rest_mass_density_expected),
                    make_not_null(&electron_fraction_expected),
                    make_not_null(&temperature_expected),
                    make_not_null(&spatial_velocity_expected),
                    make_not_null(&inverse_spatial_metric_expected), {},
                    normal_covector, normal_vector,

                    get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars),
                    get<hydro::Tags::ElectronFraction<DataVector>>(prim_vars),
                    get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
                        prim_vars),
                    get<hydro::Tags::SpatialVelocity<DataVector, 3>>(prim_vars),
                    get<hydro::Tags::MagneticField<DataVector, 3>>(prim_vars),
                    get<hydro::Tags::LorentzFactor<DataVector>>(prim_vars),
                    get<hydro::Tags::Pressure<DataVector>>(prim_vars),
                    get<hydro::Tags::Temperature<DataVector>>(prim_vars),

                    shift, lapse, inverse_spatial_metric)
                .has_value());

  tmpl::for_each<typename Vars::tags_list>([&expected_vars, &vars](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    if constexpr (not tmpl::list_contains_v<tags_not_set_by_boundary_condition,
                                            tag>) {
      CAPTURE(db::tag_name<tag>());
      CHECK(get<tag>(vars) == get<tag>(expected_vars));
    }
  });

  // Test constraint-preserving BC
  DtVars expected_dt_vars{num_points, 0.0};
  gh::BoundaryConditions::ConstraintPreservingBjorhus<3> gh_cp{
      gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType::
          ConstraintPreservingPhysical};
  CHECK(
      not gh_cp
              .dg_time_derivative(
                  make_not_null(
                      &get<
                          ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
                          expected_dt_vars)),
                  make_not_null(&get<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>(
                      expected_dt_vars)),
                  make_not_null(&get<::Tags::dt<gh::Tags::Phi<DataVector, 3>>>(
                      expected_dt_vars)),

                  {}, normal_covector, normal_vector,

                  get<gr::Tags::SpacetimeMetric<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Pi<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Phi<DataVector, 3>>(expected_vars),

                  coords, interior_gamma1, interior_gamma2,
                  get<gr::Tags::Lapse<DataVector>>(expected_vars),
                  get<gr::Tags::Shift<DataVector, 3>>(expected_vars),
                  inverse_spacetime_metric, spacetime_normal_vector,
                  three_index_constraint, gauge_source,
                  spacetime_deriv_gauge_source, logical_dt_spacetime_metric,
                  logical_dt_pi, logical_dt_phi, d_spacetime_metric, d_pi,
                  d_phi)
              .has_value());
  tmpl::for_each<typename DtVars::tags_list>(
      [&dt_vars, &expected_dt_vars](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CAPTURE(db::tag_name<tag>());
        CHECK(get<tag>(dt_vars) == get<tag>(expected_dt_vars));
      });
}

template <typename U>
void test_fd(const U& boundary_condition) {
  const auto direction = Direction<3>::lower_xi();

  grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim reconstructor{};

  using Vars = Variables<grmhd::GhValenciaDivClean::Tags::
                             primitive_grmhd_and_spacetime_reconstruction_tags>;
  Vars vars{reconstructor.ghost_zone_size() * 5};
  auto& [rest_mass_density, electron_fraction, pressure,
         lorentz_factor_times_spatial_velocity, magnetic_field,
         divergence_cleaning_field, spacetime_metric, pi, phi] = vars;

  CHECK_THROWS_WITH(
      boundary_condition.fd_ghost(
          make_not_null(&spacetime_metric), make_not_null(&pi),
          make_not_null(&phi), make_not_null(&rest_mass_density),
          make_not_null(&electron_fraction), make_not_null(&pressure),
          make_not_null(&lorentz_factor_times_spatial_velocity),
          make_not_null(&magnetic_field),
          make_not_null(&divergence_cleaning_field), direction),
      Catch::Matchers::ContainsSubstring(
          "Not implemented because it's not trivial to figure out what the"));
}
}  // namespace

// clang-format off
SPECTRE_TEST_CASE(
  "Unit.GhValenciaDivClean.BoundaryConditions.ConstraintPreservingFreeOutflow",
  "[Unit][Evolution]") {
  // clang-format on
  MAKE_GENERATOR(gen);
  register_factory_classes_with_charm<Metavariables>();

  const auto product_boundary_condition =
      TestHelpers::test_creation<
          std::unique_ptr<
              grmhd::GhValenciaDivClean::BoundaryConditions::BoundaryCondition>,
          Metavariables>(
          "ConstraintPreservingFreeOutflow:\n"
          "  Type: ConstraintPreservingPhysical\n")
          ->get_clone();

  const auto serialized_and_deserialized_condition = serialize_and_deserialize(
      *dynamic_cast<grmhd::GhValenciaDivClean::BoundaryConditions::
                        ConstraintPreservingFreeOutflow*>(
          product_boundary_condition.get()));

  test_dg(make_not_null(&gen), serialized_and_deserialized_condition);
  test_fd(serialized_and_deserialized_condition);
}
