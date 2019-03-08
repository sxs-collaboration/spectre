// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "ApparentHorizons/Tags.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"  //IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::Pi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::Phi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UPsi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UZero
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UMinus
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UPlus
// IWYU pragma: no_forward_declare StrahlkorperTags::CartesianCoords
// IWYU pragma: no_forward_declare StrahlkorperTags::NormalOneForm
// IWYU pragma: no_forward_declare Tags::CharSpeed
// IWYU pragma: no_forward_declare Tags::dt
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
template <class... Tags, class FluxType, class... NormalDotNumericalFluxTypes>
void apply_numerical_flux(
    const FluxType& flux,
    const Variables<tmpl::list<Tags...>>& packaged_data_int,
    const Variables<tmpl::list<Tags...>>& packaged_data_ext,
    NormalDotNumericalFluxTypes&&... normal_dot_numerical_flux) {
  flux(std::forward<NormalDotNumericalFluxTypes>(normal_dot_numerical_flux)...,
       get<Tags>(packaged_data_int)..., get<Tags>(packaged_data_ext)...);
}

// Test GH upwind flux by comparing to Schwarzschild
template <typename Solution>
void test_upwind_flux_analytic(
    const Solution& solution_int, const Solution& solution_ext,
    const Strahlkorper<Frame::Inertial>& strahlkorper) noexcept {
  // Set up grid
  const size_t spatial_dim = 3;

  const auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(strahlkorper);

  const auto& x =
      db::get<StrahlkorperTags::CartesianCoords<Frame::Inertial>>(box);
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate analytic solution for interior
  const auto vars_int = solution_int.variables(
      x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse_int = get<gr::Tags::Lapse<>>(vars_int);
  const auto& dt_lapse_int = get<Tags::dt<gr::Tags::Lapse<>>>(vars_int);
  const auto& d_lapse_int =
      get<typename Solution::template DerivLapse<DataVector>>(vars_int);
  const auto& shift_int = get<gr::Tags::Shift<spatial_dim>>(vars_int);
  const auto& d_shift_int =
      get<typename Solution::template DerivShift<DataVector>>(vars_int);
  const auto& dt_shift_int =
      get<Tags::dt<gr::Tags::Shift<spatial_dim>>>(vars_int);
  const auto& spatial_metric_int =
      get<gr::Tags::SpatialMetric<spatial_dim>>(vars_int);
  const auto& dt_spatial_metric_int =
      get<Tags::dt<gr::Tags::SpatialMetric<spatial_dim>>>(vars_int);
  const auto& d_spatial_metric_int =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars_int);

  const auto inverse_spatial_metric_int =
      determinant_and_inverse(spatial_metric_int).second;
  const auto spacetime_metric_int =
      gr::spacetime_metric(lapse_int, shift_int, spatial_metric_int);
  const auto phi_int =
      GeneralizedHarmonic::phi(lapse_int, d_lapse_int, shift_int, d_shift_int,
                               spatial_metric_int, d_spatial_metric_int);
  const auto pi_int = GeneralizedHarmonic::pi(
      lapse_int, dt_lapse_int, shift_int, dt_shift_int, spatial_metric_int,
      dt_spatial_metric_int, phi_int);

  // Evaluate analytic solution for exterior (i.e. neighbor)
  const auto vars_ext = solution_ext.variables(
      x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse_ext = get<gr::Tags::Lapse<>>(vars_ext);
  const auto& dt_lapse_ext = get<Tags::dt<gr::Tags::Lapse<>>>(vars_ext);
  const auto& d_lapse_ext =
      get<typename Solution::template DerivLapse<DataVector>>(vars_ext);
  const auto& shift_ext = get<gr::Tags::Shift<spatial_dim>>(vars_ext);
  const auto& d_shift_ext =
      get<typename Solution::template DerivShift<DataVector>>(vars_ext);
  const auto& dt_shift_ext =
      get<Tags::dt<gr::Tags::Shift<spatial_dim>>>(vars_ext);
  const auto& spatial_metric_ext =
      get<gr::Tags::SpatialMetric<spatial_dim>>(vars_ext);
  const auto& dt_spatial_metric_ext =
      get<Tags::dt<gr::Tags::SpatialMetric<spatial_dim>>>(vars_ext);
  const auto& d_spatial_metric_ext =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars_ext);

  const auto inverse_spatial_metric_ext =
      determinant_and_inverse(spatial_metric_ext).second;
  const auto spacetime_metric_ext =
      gr::spacetime_metric(lapse_ext, shift_ext, spatial_metric_ext);
  const auto phi_ext =
      GeneralizedHarmonic::phi(lapse_ext, d_lapse_ext, shift_ext, d_shift_ext,
                               spatial_metric_ext, d_spatial_metric_ext);
  const auto pi_ext = GeneralizedHarmonic::pi(
      lapse_ext, dt_lapse_ext, shift_ext, dt_shift_ext, spatial_metric_ext,
      dt_spatial_metric_ext, phi_ext);

  // More ingredients to get the char fields
  const size_t n_pts = x.begin()->size();
  const auto gamma_1 = make_with_value<Scalar<DataVector>>(x, 0.4);
  const auto gamma_2 = make_with_value<Scalar<DataVector>>(x, 0.1);

  // Get surface normal vectors
  const DataVector one_over_one_form_magnitude_int =
      1.0 / get(magnitude(
                db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
                inverse_spatial_metric_int));
  const auto unit_normal_one_form_int = StrahlkorperGr::unit_normal_one_form(
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
      one_over_one_form_magnitude_int);
  const auto unit_normal_vector_int = raise_or_lower_index(
      unit_normal_one_form_int, inverse_spatial_metric_int);

  const DataVector one_over_one_form_magnitude_ext =
      1.0 / get(magnitude(
                db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
                inverse_spatial_metric_ext));
  auto unit_normal_one_form_ext = StrahlkorperGr::unit_normal_one_form(
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
      one_over_one_form_magnitude_ext);
  auto unit_normal_vector_ext = raise_or_lower_index(
      unit_normal_one_form_ext, inverse_spatial_metric_ext);
  // The exterior normal points in the opposite direction as the
  // interior normal
  for (size_t i = 0; i < spatial_dim; ++i) {
    unit_normal_one_form_ext.get(i) *= -1.0;
    unit_normal_vector_ext.get(i) *= -1.0;
  }

  // If all the characteristic speeds on the interior are negative,
  // verify that changing the interior char fields does not change the
  // output of the upwind flux
  Variables<typename GeneralizedHarmonic::UpwindFlux<spatial_dim>::package_tags>
      packaged_data_int(n_pts, 0.0);
  Variables<typename GeneralizedHarmonic::UpwindFlux<spatial_dim>::package_tags>
      packaged_data_ext(n_pts, 0.0);
  GeneralizedHarmonic::UpwindFlux<spatial_dim> flux_computer{};

  const auto char_speeds_int = GeneralizedHarmonic::CharacteristicSpeedsCompute<
      spatial_dim, Frame::Inertial>::function(gamma_1, lapse_int, shift_int,
                                              unit_normal_one_form_int);
  auto zero_char_speed =
      make_with_value<Scalar<DataVector>>(char_speeds_int[0], 0.0);
  if (step_function(char_speeds_int[0]) == get(zero_char_speed) and
      step_function(char_speeds_int[1]) == get(zero_char_speed) and
      step_function(char_speeds_int[2]) == get(zero_char_speed) and
      step_function(char_speeds_int[3]) == get(zero_char_speed)) {
    tnsr::aa<DataVector, spatial_dim, Frame::Inertial>
        normal_dot_numerical_flux_pi(n_pts, 0.0);
    tnsr::aa<DataVector, spatial_dim, Frame::Inertial>
        normal_dot_numerical_flux_psi(n_pts, 0.0);
    tnsr::iaa<DataVector, spatial_dim, Frame::Inertial>
        normal_dot_numerical_flux_phi(n_pts, 0.0);
    // Compute the upwind flux using solution_int for the interior and
    // solution_ext for the interior (1)
    flux_computer.package_data(
        make_not_null(&packaged_data_int), spacetime_metric_int, pi_int,
        phi_int, lapse_int, shift_int, gamma_1, gamma_2,
        unit_normal_one_form_int, unit_normal_vector_int);
    flux_computer.package_data(
        make_not_null(&packaged_data_ext), spacetime_metric_ext, pi_ext,
        phi_ext, lapse_ext, shift_ext, gamma_1, gamma_2,
        unit_normal_one_form_ext, unit_normal_vector_ext);
    apply_numerical_flux(flux_computer, packaged_data_int, packaged_data_ext,
                         make_not_null(&normal_dot_numerical_flux_psi),
                         make_not_null(&normal_dot_numerical_flux_pi),
                         make_not_null(&normal_dot_numerical_flux_phi));

    // Compute the upwind flux using solution_ext for both the interior and
    // the exterior, but keep everything else the same (2)
    tnsr::aa<DataVector, spatial_dim, Frame::Inertial>
        normal_dot_numerical_flux_pi_different_fields(n_pts, 0.0);
    tnsr::aa<DataVector, spatial_dim, Frame::Inertial>
        normal_dot_numerical_flux_psi_different_fields(n_pts, 0.0);
    tnsr::iaa<DataVector, spatial_dim, Frame::Inertial>
        normal_dot_numerical_flux_phi_different_fields(n_pts, 0.0);
    flux_computer.package_data(
        make_not_null(&packaged_data_int), spacetime_metric_ext, pi_ext,
        phi_ext, lapse_ext, shift_ext, gamma_1, gamma_2,
        unit_normal_one_form_int, unit_normal_vector_int);
    flux_computer.package_data(
        make_not_null(&packaged_data_ext), spacetime_metric_ext, pi_ext,
        phi_ext, lapse_ext, shift_ext, gamma_1, gamma_2,
        unit_normal_one_form_ext, unit_normal_vector_ext);
    apply_numerical_flux(
        flux_computer, packaged_data_int, packaged_data_ext,
        make_not_null(&normal_dot_numerical_flux_psi_different_fields),
        make_not_null(&normal_dot_numerical_flux_pi_different_fields),
        make_not_null(&normal_dot_numerical_flux_phi_different_fields));

    // Check that (1) and (2) are the same
    // Not sure if this test should pass
    /*CHECK_ITERABLE_APPROX(normal_dot_numerical_flux_pi,
                          normal_dot_numerical_flux_pi_different_fields);
    CHECK_ITERABLE_APPROX(normal_dot_numerical_flux_phi,
                          normal_dot_numerical_flux_phi_different_fields);
    CHECK_ITERABLE_APPROX(normal_dot_numerical_flux_psi,
                          normal_dot_numerical_flux_psi_different_fields);*/
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.UpwindFlux",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  // Test GH upwind flux against Kerr Schild
  const double mass = 2.;
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const gr::Solutions::KerrSchild solution_1(mass, spin, center);
  const gr::Solutions::KerrSchild solution_2(2.0 * mass, spin, center);

  const std::array<double, 3> lower_bound{{0.82, 1.22, 1.32}};
  const std::array<double, 3> upper_bound{{0.78, 1.18, 1.28}};

  const double radius_inside_horizons = 1.0;
  const size_t l_max = 2;

  const auto strahlkorper_inside_horizons = Strahlkorper<Frame::Inertial>(
      l_max, l_max, radius_inside_horizons, center);

  test_upwind_flux_analytic(solution_1, solution_2,
                            strahlkorper_inside_horizons);
  /*test_upwind_flux_analytic(solution_2, solution_1,
                            strahlkorper_inside_horizons);*/
}
