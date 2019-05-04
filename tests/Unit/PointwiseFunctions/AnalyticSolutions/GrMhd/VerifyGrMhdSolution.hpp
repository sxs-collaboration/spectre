// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace VerifyGrMhdSolution_detail {

using valencia_tags = tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                                 grmhd::ValenciaDivClean::Tags::TildeTau,
                                 grmhd::ValenciaDivClean::Tags::TildeS<>,
                                 grmhd::ValenciaDivClean::Tags::TildeB<>,
                                 grmhd::ValenciaDivClean::Tags::TildePhi>;

// compute the time derivative using a centered sixth-order stencil
template <typename Solution>
Variables<valencia_tags> numerical_dt(
    const Solution& solution, const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    const double time, const double delta_time) {
  std::array<double, 6> six_times{
      {time - 3.0 * delta_time, time - 2.0 * delta_time, time - delta_time,
       time + delta_time, time + 2.0 * delta_time, time + 3.0 * delta_time}};
  const size_t number_of_points = get<0>(x).size();
  auto solution_at_six_times =
      make_array<6>(Variables<valencia_tags>(number_of_points));

  using solution_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
                 hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>>;

  for (size_t i = 0; i < 6; ++i) {
    const auto vars =
        solution.variables(x, gsl::at(six_times, i), solution_tags{});

    const auto& rest_mass_density =
        get<hydro::Tags::RestMassDensity<DataVector>>(vars);
    const auto& specific_internal_energy =
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(vars);
    const auto& spatial_velocity =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(vars);
    const auto& magnetic_field =
        get<hydro::Tags::MagneticField<DataVector, 3>>(vars);
    const auto& divergence_cleaning_field =
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(vars);
    const auto& lorentz_factor =
        get<hydro::Tags::LorentzFactor<DataVector>>(vars);
    const auto& pressure = get<hydro::Tags::Pressure<DataVector>>(vars);
    const auto& specific_enthalpy =
        get<hydro::Tags::SpecificEnthalpy<DataVector>>(vars);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(vars);
    const auto& sqrt_det_spatial_metric =
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars);

    grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
        make_not_null(&get<grmhd::ValenciaDivClean::Tags::TildeD>(
            gsl::at(solution_at_six_times, i))),
        make_not_null(&get<grmhd::ValenciaDivClean::Tags::TildeTau>(
            gsl::at(solution_at_six_times, i))),
        make_not_null(&get<grmhd::ValenciaDivClean::Tags::TildeS<>>(
            gsl::at(solution_at_six_times, i))),
        make_not_null(&get<grmhd::ValenciaDivClean::Tags::TildeB<>>(
            gsl::at(solution_at_six_times, i))),
        make_not_null(&get<grmhd::ValenciaDivClean::Tags::TildePhi>(
            gsl::at(solution_at_six_times, i))),
        rest_mass_density, specific_internal_energy, specific_enthalpy,
        pressure, spatial_velocity, lorentz_factor, magnetic_field,
        sqrt_det_spatial_metric, spatial_metric, divergence_cleaning_field);
  }

  return (-1.0 / (60.0 * delta_time)) * solution_at_six_times[0] +
         (3.0 / (20.0 * delta_time)) * solution_at_six_times[1] +
         (-0.75 / delta_time) * solution_at_six_times[2] +
         (0.75 / delta_time) * solution_at_six_times[3] +
         (-3.0 / (20.0 * delta_time)) * solution_at_six_times[4] +
         (1.0 / (60.0 * delta_time)) * solution_at_six_times[5];
}
}  // namespace VerifyGrMhdSolution_detail

/// \ingroup TestingFrameworkGroup
/// \brief Determines if the given `solution` is a solution of the GRMHD
/// equations.
///
/// Uses numerical derivatives to compute the solution, on the given `mesh` of
/// the root Element of the given `block` at the given `time` using a
/// sixth-order derivative in time for the given `delta_time`. The maximum
/// residual of the GRMHD equations must be zero within `error_tolerance`
template <typename Solution>
void verify_grmhd_solution(const Solution& solution,
                           const Block<3, Frame::Inertial>& block,
                           const Mesh<3>& mesh, const double error_tolerance,
                           const double time,
                           const double delta_time) noexcept {
  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = block.coordinate_map()(x_logical);

  // Evaluate analytic solution
  using solution_tags = tmpl::list<
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpecificInternalEnergy<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
      hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>,
      hydro::Tags::DivergenceCleaningField<DataVector>,
      hydro::Tags::LorentzFactor<DataVector>, hydro::Tags::Pressure<DataVector>,
      hydro::Tags::SpecificEnthalpy<DataVector>, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<3, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>;
  const auto vars = solution.variables(x, time, solution_tags{});

  const auto& rest_mass_density =
      get<hydro::Tags::RestMassDensity<DataVector>>(vars);
  const auto& specific_internal_energy =
      get<hydro::Tags::SpecificInternalEnergy<DataVector>>(vars);
  const auto& spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(vars);
  const auto& magnetic_field =
      get<hydro::Tags::MagneticField<DataVector, 3>>(vars);
  const auto& divergence_cleaning_field =
      get<hydro::Tags::DivergenceCleaningField<DataVector>>(vars);
  const auto& lorentz_factor =
      get<hydro::Tags::LorentzFactor<DataVector>>(vars);
  const auto& pressure = get<hydro::Tags::Pressure<DataVector>>(vars);
  const auto& specific_enthalpy =
      get<hydro::Tags::SpecificEnthalpy<DataVector>>(vars);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& d_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(vars);
  const auto& shift =
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(vars);
  const auto& d_shift =
      get<::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(vars);
  const auto& d_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(vars);
  const auto& inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>(vars);
  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars);
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>(vars);

  const size_t number_of_points = mesh.number_of_grid_points();
  Scalar<DataVector> tilde_d(number_of_points);
  Scalar<DataVector> tilde_tau(number_of_points);
  tnsr::i<DataVector, 3> tilde_s(number_of_points);
  tnsr::I<DataVector, 3> tilde_b(number_of_points);
  Scalar<DataVector> tilde_phi(number_of_points);

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&tilde_d), make_not_null(&tilde_tau),
      make_not_null(&tilde_s), make_not_null(&tilde_b),
      make_not_null(&tilde_phi), rest_mass_density, specific_internal_energy,
      specific_enthalpy, pressure, spatial_velocity, lorentz_factor,
      magnetic_field, sqrt_det_spatial_metric, spatial_metric,
      divergence_cleaning_field);

  using flux_tags =
      tmpl::list<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD,
                            tmpl::size_t<3>, Frame::Inertial>,
                 Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau,
                            tmpl::size_t<3>, Frame::Inertial>,
                 Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeS<>,
                            tmpl::size_t<3>, Frame::Inertial>,
                 Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeB<>,
                            tmpl::size_t<3>, Frame::Inertial>,
                 Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi,
                            tmpl::size_t<3>, Frame::Inertial>>;
  Variables<flux_tags> fluxes(number_of_points);
  auto& flux_tilde_d =
      get<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD, tmpl::size_t<3>,
                     Frame::Inertial>>(fluxes);
  auto& flux_tilde_tau =
      get<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau, tmpl::size_t<3>,
                     Frame::Inertial>>(fluxes);
  auto& flux_tilde_s =
      get<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeS<>, tmpl::size_t<3>,
                     Frame::Inertial>>(fluxes);
  auto& flux_tilde_b =
      get<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeB<>, tmpl::size_t<3>,
                     Frame::Inertial>>(fluxes);
  auto& flux_tilde_phi =
      get<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi, tmpl::size_t<3>,
                     Frame::Inertial>>(fluxes);

  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&flux_tilde_d), make_not_null(&flux_tilde_tau),
      make_not_null(&flux_tilde_s), make_not_null(&flux_tilde_b),
      make_not_null(&flux_tilde_phi), tilde_d, tilde_tau, tilde_s, tilde_b,
      tilde_phi, lapse, shift, sqrt_det_spatial_metric, spatial_metric,
      inv_spatial_metric, pressure, spatial_velocity, lorentz_factor,
      magnetic_field);

  const auto div_of_fluxes = divergence<flux_tags, 3, Frame::Inertial>(
      fluxes, mesh, block.coordinate_map().inv_jacobian(x_logical));

  const auto& div_flux_tilde_d =
      get<Tags::div<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD,
                               tmpl::size_t<3>, Frame::Inertial>>>(
          div_of_fluxes);
  const auto& div_flux_tilde_tau =
      get<Tags::div<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau,
                               tmpl::size_t<3>, Frame::Inertial>>>(
          div_of_fluxes);
  const auto& div_flux_tilde_s =
      get<Tags::div<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeS<>,
                               tmpl::size_t<3>, Frame::Inertial>>>(
          div_of_fluxes);
  const auto& div_flux_tilde_b =
      get<Tags::div<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeB<>,
                               tmpl::size_t<3>, Frame::Inertial>>>(
          div_of_fluxes);
  const auto& div_flux_tilde_phi =
      get<Tags::div<Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi,
                               tmpl::size_t<3>, Frame::Inertial>>>(
          div_of_fluxes);

  Scalar<DataVector> source_tilde_tau(number_of_points);
  tnsr::i<DataVector, 3> source_tilde_s(number_of_points);
  tnsr::I<DataVector, 3> source_tilde_b(number_of_points);
  Scalar<DataVector> source_tilde_phi(number_of_points);
  grmhd::ValenciaDivClean::ComputeSources::apply(
      make_not_null(&source_tilde_tau), make_not_null(&source_tilde_s),
      make_not_null(&source_tilde_b), make_not_null(&source_tilde_phi), tilde_d,
      tilde_tau, tilde_s, tilde_b, tilde_phi, spatial_velocity, magnetic_field,
      rest_mass_density, specific_enthalpy, lorentz_factor, pressure, lapse,
      d_lapse, d_shift, spatial_metric, d_spatial_metric, inv_spatial_metric,
      sqrt_det_spatial_metric, extrinsic_curvature, 0.0);

  Scalar<DataVector> residual_tilde_d(number_of_points, 0.);
  Scalar<DataVector> residual_tilde_tau(number_of_points, 0.);
  tnsr::i<DataVector, 3> residual_tilde_s(number_of_points, 0.);
  tnsr::I<DataVector, 3> residual_tilde_b(number_of_points, 0.);
  Scalar<DataVector> residual_tilde_phi(number_of_points, 0.);

  Variables<VerifyGrMhdSolution_detail::valencia_tags> dt_solution =
      VerifyGrMhdSolution_detail::numerical_dt(solution, x, time, delta_time);

  get(residual_tilde_d) =
      get(get<grmhd::ValenciaDivClean::Tags::TildeD>(dt_solution)) +
      get(div_flux_tilde_d);
  get(residual_tilde_tau) =
      get(get<grmhd::ValenciaDivClean::Tags::TildeTau>(dt_solution)) +
      get(div_flux_tilde_tau) - get(source_tilde_tau);
  get<0>(residual_tilde_s) =
      get<0>(get<grmhd::ValenciaDivClean::Tags::TildeS<>>(dt_solution)) +
      get<0>(div_flux_tilde_s) - get<0>(source_tilde_s);
  get<1>(residual_tilde_s) =
      get<1>(get<grmhd::ValenciaDivClean::Tags::TildeS<>>(dt_solution)) +
      get<1>(div_flux_tilde_s) - get<1>(source_tilde_s);
  get<2>(residual_tilde_s) =
      get<2>(get<grmhd::ValenciaDivClean::Tags::TildeS<>>(dt_solution)) +
      get<2>(div_flux_tilde_s) - get<2>(source_tilde_s);
  get<0>(residual_tilde_b) =
      get<0>(get<grmhd::ValenciaDivClean::Tags::TildeB<>>(dt_solution)) +
      get<0>(div_flux_tilde_b) - get<0>(source_tilde_b);
  get<1>(residual_tilde_b) =
      get<1>(get<grmhd::ValenciaDivClean::Tags::TildeB<>>(dt_solution)) +
      get<1>(div_flux_tilde_b) - get<1>(source_tilde_b);
  get<2>(residual_tilde_b) =
      get<2>(get<grmhd::ValenciaDivClean::Tags::TildeB<>>(dt_solution)) +
      get<2>(div_flux_tilde_b) - get<2>(source_tilde_b);
  get(residual_tilde_phi) +=
      get(get<grmhd::ValenciaDivClean::Tags::TildePhi>(dt_solution)) +
      get(div_flux_tilde_phi) - get(source_tilde_phi);

  Approx numerical_approx =
      Approx::custom().epsilon(error_tolerance).scale(1.0);
  CHECK(max(abs(get(residual_tilde_d))) == numerical_approx(0.0));
  CHECK(max(abs(get(residual_tilde_tau))) == numerical_approx(0.0));
  CHECK(max(abs(get<0>(residual_tilde_s))) == numerical_approx(0.0));
  CHECK(max(abs(get<1>(residual_tilde_s))) == numerical_approx(0.0));
  CHECK(max(abs(get<2>(residual_tilde_s))) == numerical_approx(0.0));
  CHECK(max(abs(get<0>(residual_tilde_b))) == numerical_approx(0.0));
  CHECK(max(abs(get<1>(residual_tilde_b))) == numerical_approx(0.0));
  CHECK(max(abs(get<2>(residual_tilde_b))) == numerical_approx(0.0));
  CHECK(max(abs(get(residual_tilde_phi))) == numerical_approx(0.0));
}
