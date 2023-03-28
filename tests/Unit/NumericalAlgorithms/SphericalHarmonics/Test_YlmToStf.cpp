// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/RealSphericalHarmonics.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/YlmToStf.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.YlmToStf",
                  "[NumericalAlgorithms][Unit]") {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> coef_distribution(-0.1, 0.1);

  std::uniform_real_distribution<double> theta_distribution(0., M_PI);
  std::uniform_real_distribution<double> phi_distribution(0., 2 * M_PI);

  const size_t num_points = 10000;

  const auto theta = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&theta_distribution),
      DataVector(num_points));
  const auto phi = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&phi_distribution),
      DataVector(num_points));

  tnsr::I<DataVector, 3> normal_vector(num_points);
  get<0>(normal_vector) = sin(theta) * cos(phi);
  get<1>(normal_vector) = sin(theta) * sin(phi);
  get<2>(normal_vector) = cos(theta);

  {
    // l=0
    const auto ylm_l0_coefs = make_with_random_values<ModalVector>(
        make_not_null(&generator), make_not_null(&coef_distribution),
        ModalVector(1));
    const auto stf_l0_coefs = ylm_to_stf_0(ylm_l0_coefs);

    DataVector stf_l0_at_points(num_points, get(stf_l0_coefs));
    const DataVector ylm_l0_at_points =
        ylm_l0_coefs.at(0) * real_spherical_harmonic(theta, phi, 0, 0);
    CHECK_ITERABLE_APPROX(stf_l0_at_points, ylm_l0_at_points);
  }
  {
    // l=1
    const auto ylm_l1_coefs = make_with_random_values<ModalVector>(
        make_not_null(&generator), make_not_null(&coef_distribution),
        ModalVector(3));
    const auto stf_l1_coefs = ylm_to_stf_1<Frame::Grid>(ylm_l1_coefs);

    DataVector stf_l1_at_points(num_points, 0.);
    DataVector ylm_l1_at_points(num_points, 0.);
    for (size_t i = 0; i < 3; ++i) {
      stf_l1_at_points += stf_l1_coefs.get(i) * normal_vector.get(i);
      ylm_l1_at_points +=
          ylm_l1_coefs.at(i) * real_spherical_harmonic(theta, phi, 1, i - 1);
    }
    CHECK_ITERABLE_APPROX(stf_l1_at_points, ylm_l1_at_points);
  }
  {
    // l=2
    const auto ylm_l2_coefs = make_with_random_values<ModalVector>(
        make_not_null(&generator), make_not_null(&coef_distribution),
        ModalVector(5));
    const auto stf_l2_coefs = ylm_to_stf_2<Frame::Grid>(ylm_l2_coefs);

    // check trace is zero
    CHECK((stf_l2_coefs.get(0, 0) + stf_l2_coefs.get(1, 1) +
           stf_l2_coefs.get(2, 2)) == approx(0.));
    DataVector stf_l2_at_points(num_points, 0.);
    DataVector ylm_l2_at_points(num_points, 0.);
    for (size_t i = 0; i < 5; ++i) {
      ylm_l2_at_points +=
          ylm_l2_coefs.at(i) * real_spherical_harmonic(theta, phi, 2, i - 2);
    }

    // we don't need to compute the trace-free part of the second order normal
    // vector combination because the trace is automatically removed by
    // contracting with the trace-free coefficient tensor.
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        stf_l2_at_points += stf_l2_coefs.get(i, j) * normal_vector.get(i) *
                            normal_vector.get(j);
      }
    }
    CHECK_ITERABLE_APPROX(stf_l2_at_points, ylm_l2_at_points);
  }
}
