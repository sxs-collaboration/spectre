// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/RandomUnitNormal.hpp"
#include "Helpers/NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace CurvedScalarWave::CurvedScalarWave_detail {
template <size_t Dim>
using char_field_tags =
    tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>;

template <size_t Dim>
Variables<char_field_tags<Dim>> weight_char_fields(
    const Variables<char_field_tags<Dim>>& char_fields_int,
    const std::array<DataVector, 4>& char_speeds_int,
    const Variables<char_field_tags<Dim>>& char_fields_ext,
    const std::array<DataVector, 4>& char_speeds_ext) noexcept;
}  // namespace CurvedScalarWave::CurvedScalarWave_detail

namespace {
// Test CSW upwind flux using random fields
template <size_t Dim>
void test_upwind_flux_random() noexcept {
  static_assert(tt::assert_conforms_to<CurvedScalarWave::UpwindFlux<Dim>,
                                       dg::protocols::NumericalFlux>);

  const DataVector used_for_size{5,
                                 std::numeric_limits<double>::signaling_NaN()};

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  std::uniform_real_distribution<> dist_pert(-0.1, 0.1);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);
  const auto nn_dist_pert = make_not_null(&dist_pert);

  const auto one = make_with_value<Scalar<DataVector>>(used_for_size, 1.);
  const auto minus_one =
      make_with_value<Scalar<DataVector>>(used_for_size, -1.);
  const auto five = make_with_value<Scalar<DataVector>>(used_for_size, 5.);
  const auto minus_five =
      make_with_value<Scalar<DataVector>>(used_for_size, -5.);

  // Set psi, pi, phi to be random (phi, pi should not need to be consistent
  // with psi for the flux consistency tests to pass)
  const auto psi_int = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist_pert, used_for_size);
  const auto psi_ext = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist_pert, used_for_size);
  const auto phi_int =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_dist_pert, used_for_size);
  const auto phi_ext =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_dist_pert, used_for_size);
  const auto pi_int = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist_pert, used_for_size);
  const auto pi_ext = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist_pert, used_for_size);

  // Choose spacetime metric randomly, but make sure the result is
  // still invertible. To do this, start with
  // Minkowski, and then add a 10% random perturbation.
  auto spacetime_metric_int =
      make_with_random_values<tnsr::aa<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_dist_pert, used_for_size);
  get<0, 0>(spacetime_metric_int) += get(minus_one);
  for (size_t i = 1; i <= Dim; ++i) {
    spacetime_metric_int.get(i, i) += get(one);
  }

  auto spacetime_metric_ext =
      make_with_random_values<tnsr::aa<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_dist_pert, used_for_size);
  get<0, 0>(spacetime_metric_ext) += get(minus_one);
  for (size_t i = 1; i <= Dim; ++i) {
    spacetime_metric_ext.get(i, i) += get(one);
  }

  const auto spatial_metric_int = gr::spatial_metric(spacetime_metric_int);
  const auto inverse_spatial_metric_int =
      determinant_and_inverse(spatial_metric_int).second;
  const tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form_int =
      raise_or_lower_index(random_unit_normal(nn_generator, spatial_metric_int),
                           spatial_metric_int);
  tnsr::i<DataVector, Dim, Frame::Inertial> minus_unit_normal_one_form_int =
      unit_normal_one_form_int;
  for (size_t i = 0; i < Dim; ++i) {
    minus_unit_normal_one_form_int.get(i) *= -1.;
  }

  const auto shift_int =
      gr::shift(spacetime_metric_int, inverse_spatial_metric_int);
  const auto lapse_int = gr::lapse(shift_int, spacetime_metric_int);

  const auto spatial_metric_ext = gr::spatial_metric(spacetime_metric_ext);
  const auto inverse_spatial_metric_ext =
      determinant_and_inverse(spatial_metric_ext).second;
  const DataVector one_form_magnitude_ext =
      get(magnitude(unit_normal_one_form_int, inverse_spatial_metric_ext));

  const auto shift_ext =
      gr::shift(spacetime_metric_ext, inverse_spatial_metric_ext);
  const auto lapse_ext = gr::lapse(shift_int, spacetime_metric_ext);

  const auto gamma_1 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, used_for_size);
  const auto gamma_2 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, used_for_size);

  // Get the characteristic fields and speeds
  const auto char_fields_int = CurvedScalarWave::characteristic_fields(
      gamma_2, inverse_spatial_metric_int, psi_int, pi_int, phi_int,
      unit_normal_one_form_int);
  const auto char_fields_ext = CurvedScalarWave::characteristic_fields(
      gamma_2, inverse_spatial_metric_ext, psi_ext, pi_ext, phi_ext,
      unit_normal_one_form_int);

  std::array<DataVector, 4> char_speeds_one{
      {get(one), get(one), get(one), get(one)}};
  std::array<DataVector, 4> char_speeds_minus_one{
      {get(minus_one), get(minus_one), get(minus_one), get(minus_one)}};
  std::array<DataVector, 4> char_speeds_five{
      {get(five), get(five), get(five), get(five)}};
  std::array<DataVector, 4> char_speeds_minus_five{
      {get(minus_five), get(minus_five), get(minus_five), get(minus_five)}};

  INFO("test curved-scalar-wave upwind weighting function")
  // If all the char speeds are +1, the weighted fields should just
  // be the interior fields
  const auto weighted_char_fields_one =
      CurvedScalarWave::CurvedScalarWave_detail::weight_char_fields<Dim>(
          char_fields_int, char_speeds_one, char_fields_ext, char_speeds_one);
  CHECK(weighted_char_fields_one == char_fields_int);

  // If all the char speeds are -1, the weighted fields should just be
  // the exterior fields up to a sign
  const auto weighted_char_fields_minus_one =
      CurvedScalarWave::CurvedScalarWave_detail::weight_char_fields<Dim>(
          char_fields_int, char_speeds_minus_one, char_fields_ext,
          char_speeds_minus_one);
  CHECK(weighted_char_fields_minus_one == -1. * char_fields_ext);

  // Check scaling by 5 instead of 1
  const auto weighted_char_fields_minus_five =
      CurvedScalarWave::CurvedScalarWave_detail::weight_char_fields<Dim>(
          char_fields_int, char_speeds_minus_five, char_fields_ext,
          char_speeds_minus_five);
  const auto weighted_char_fields_five =
      CurvedScalarWave::CurvedScalarWave_detail::weight_char_fields<Dim>(
          char_fields_int, char_speeds_five, char_fields_ext, char_speeds_five);
  CHECK(weighted_char_fields_minus_five == -5. * char_fields_ext);
  CHECK(weighted_char_fields_five == 5. * char_fields_int);

  // Check that if the same fields are given for the interior and exterior
  // (except that the normal vector gets multiplied by -1.0) that the
  // numerical flux reduces to the flux
  INFO("test consistency of the curved-scalar-wave upwind flux")
  CurvedScalarWave::UpwindFlux<Dim> flux_computer{};
  auto packaged_data_int = ::TestHelpers::NumericalFluxes::get_packaged_data(
      flux_computer, used_for_size, pi_int, phi_int, psi_int, lapse_int,
      shift_int, inverse_spatial_metric_int, gamma_1, gamma_2,
      unit_normal_one_form_int);

  auto packaged_data_int_opposite_normal =
      ::TestHelpers::NumericalFluxes::get_packaged_data(
          flux_computer, used_for_size, pi_int, phi_int, psi_int, lapse_int,
          shift_int, inverse_spatial_metric_int, gamma_1, gamma_2,
          minus_unit_normal_one_form_int);

  auto psi_normal_dot_numerical_flux = make_with_value<
      db::item_type<::Tags::NormalDotNumericalFlux<CurvedScalarWave::Psi>>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto pi_normal_dot_numerical_flux = make_with_value<
      db::item_type<::Tags::NormalDotNumericalFlux<CurvedScalarWave::Pi>>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto phi_normal_dot_numerical_flux = make_with_value<db::item_type<
      ::Tags::NormalDotNumericalFlux<CurvedScalarWave::Phi<Dim>>>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      flux_computer, packaged_data_int, packaged_data_int_opposite_normal,
      make_not_null(&pi_normal_dot_numerical_flux),
      make_not_null(&phi_normal_dot_numerical_flux),
      make_not_null(&psi_normal_dot_numerical_flux));

  CurvedScalarWave::ComputeNormalDotFluxes<Dim> normal_dot_flux_computer{};
  auto psi_normal_dot_flux = make_with_value<
      db::item_type<::Tags::NormalDotFlux<CurvedScalarWave::Psi>>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto pi_normal_dot_flux = make_with_value<
      db::item_type<::Tags::NormalDotFlux<CurvedScalarWave::Pi>>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto phi_normal_dot_flux = make_with_value<
      db::item_type<::Tags::NormalDotFlux<CurvedScalarWave::Phi<Dim>>>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  normal_dot_flux_computer.apply(
      make_not_null(&pi_normal_dot_flux), make_not_null(&phi_normal_dot_flux),
      make_not_null(&psi_normal_dot_flux), pi_int, phi_int, psi_int, gamma_1,
      gamma_2, lapse_int, shift_int, inverse_spatial_metric_int,
      unit_normal_one_form_int);

  CHECK_ITERABLE_APPROX(psi_normal_dot_numerical_flux, psi_normal_dot_flux);
  CHECK_ITERABLE_APPROX(pi_normal_dot_numerical_flux, pi_normal_dot_flux);
  CHECK_ITERABLE_APPROX(phi_normal_dot_numerical_flux, phi_normal_dot_flux);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.UpwindFlux",
                  "[Unit][Evolution]") {
  test_upwind_flux_random<1>();
  test_upwind_flux_random<2>();
  test_upwind_flux_random<3>();
}
