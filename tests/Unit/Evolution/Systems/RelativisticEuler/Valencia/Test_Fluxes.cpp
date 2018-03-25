// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Fluxes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::Inertial> expected_tilde_d_flux(
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        spatial_velocity) noexcept {
  auto result = make_with_value<tnsr::I<DataVector, Dim>>(lapse, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) =
        get(tilde_d) * (get(lapse) * spatial_velocity.get(i) - shift.get(i));
  }
  return result;
}

template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::Inertial> expected_tilde_tau_flux(
    const Scalar<DataVector>& tilde_tau, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        spatial_velocity) noexcept {
  auto result = make_with_value<tnsr::I<DataVector, Dim>>(lapse, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) =
        get(sqrt_det_spatial_metric) * get(lapse) * get(pressure) *
            spatial_velocity.get(i) +
        get(tilde_tau) * (get(lapse) * spatial_velocity.get(i) - shift.get(i));
  }
  return result;
}

template <size_t Dim>
tnsr::Ij<DataVector, Dim, Frame::Inertial> expected_tilde_s_flux(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        spatial_velocity) noexcept {
  auto result = make_with_value<tnsr::Ij<DataVector, Dim>>(lapse, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i, i) =
        get(sqrt_det_spatial_metric) * get(lapse) * get(pressure);
    for (size_t j = 0; j < Dim; ++j) {
      result.get(i, j) += tilde_s.get(j) *
                          (get(lapse) * spatial_velocity.get(i) - shift.get(i));
    }
  }
  return result;
}

template <size_t Dim>
void test_fluxes(
    const gsl::not_null<std::mt19937*> generator,
    const gsl::not_null<std::uniform_real_distribution<>*> distribution,
    const DataVector& used_for_size) noexcept {
  const auto tilde_d = make_with_random_values<Scalar<DataVector>>(
      generator, distribution, used_for_size);
  const auto tilde_tau = make_with_random_values<Scalar<DataVector>>(
      generator, distribution, used_for_size);
  const auto tilde_s = make_with_random_values<tnsr::i<DataVector, Dim>>(
      generator, distribution, used_for_size);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      generator, distribution, used_for_size);
  const auto pressure = make_with_random_values<Scalar<DataVector>>(
      generator, distribution, used_for_size);
  const auto sqrt_det_spatial_metric =
      make_with_random_values<Scalar<DataVector>>(generator, distribution,
                                                  used_for_size);
  const auto shift = make_with_random_values<tnsr::I<DataVector, Dim>>(
      generator, distribution, used_for_size);
  const auto spatial_velocity =
      make_with_random_values<tnsr::I<DataVector, Dim>>(generator, distribution,
                                                        used_for_size);

  auto tilde_d_flux =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.);
  auto tilde_tau_flux =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.);
  auto tilde_s_flux =
      make_with_value<tnsr::Ij<DataVector, Dim>>(used_for_size, 0.);

  RelativisticEuler::Valencia::fluxes(
      make_not_null(&tilde_d_flux), make_not_null(&tilde_tau_flux),
      make_not_null(&tilde_s_flux), tilde_d, tilde_tau, tilde_s, lapse, shift,
      sqrt_det_spatial_metric, pressure, spatial_velocity);

  CHECK_ITERABLE_APPROX(
      expected_tilde_d_flux(tilde_d, lapse, shift, spatial_velocity),
      tilde_d_flux);

  CHECK_ITERABLE_APPROX(
      expected_tilde_tau_flux(tilde_tau, lapse, shift, sqrt_det_spatial_metric,
                              pressure, spatial_velocity),
      tilde_tau_flux);

  CHECK_ITERABLE_APPROX(
      expected_tilde_s_flux(tilde_s, lapse, shift, sqrt_det_spatial_metric,
                            pressure, spatial_velocity),
      tilde_s_flux);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.Fluxes",
                  "[Unit][RelativisticEuler]") {
  std::random_device r;
  const auto seed = r();
  std::mt19937 generator(seed);
  INFO("seed = " << seed);
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const DataVector dv(5);
  test_fluxes<1>(nn_generator, nn_distribution, dv);
  test_fluxes<2>(nn_generator, nn_distribution, dv);
  test_fluxes<3>(nn_generator, nn_distribution, dv);
}
