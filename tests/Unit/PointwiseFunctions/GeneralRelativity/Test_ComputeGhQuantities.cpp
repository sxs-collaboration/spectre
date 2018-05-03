// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/PointwiseFunctions/GeneralRelativity/GrTestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {
void test_compute_1d_phi(const DataVector& used_for_size) {
  const size_t dim = 1;
  const auto phi = GeneralizedHarmonic::phi(
      make_lapse(0.), make_deriv_lapse<dim>(0.), make_shift<dim>(0.),
      make_deriv_shift<dim>(0.), make_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(phi.get(0, 0, 0) == approx(2.0));
  CHECK(phi.get(0, 0, 1) == approx(10.0));
  CHECK(phi.get(0, 1, 1) == approx(3.0));

  check_tensor_doubles_equals_tensor_datavectors(
      GeneralizedHarmonic::phi(
          make_lapse(used_for_size), make_deriv_lapse<dim>(used_for_size),
          make_shift<dim>(used_for_size), make_deriv_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_deriv_spatial_metric<dim>(used_for_size)),
      phi);
}

void test_compute_2d_phi(const DataVector& used_for_size) {
  const size_t dim = 2;
  const auto phi = GeneralizedHarmonic::phi(
      make_lapse(0.), make_deriv_lapse<dim>(0.), make_shift<dim>(0.),
      make_deriv_shift<dim>(0.), make_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(phi.get(0, 0, 0) == approx(330.0));
  CHECK(phi.get(0, 0, 1) == approx(42.0));
  CHECK(phi.get(0, 0, 2) == approx(84.0));
  CHECK(phi.get(0, 1, 1) == approx(3.0));
  CHECK(phi.get(0, 1, 2) == approx(6.0));
  CHECK(phi.get(0, 2, 2) == approx(12.0));
  CHECK(phi.get(1, 0, 0) == approx(294.0));
  CHECK(phi.get(1, 0, 1) == approx(42.0));
  CHECK(phi.get(1, 0, 2) == approx(81.0));
  CHECK(phi.get(1, 1, 1) == approx(4.0));
  CHECK(phi.get(1, 1, 2) == approx(7.0));
  CHECK(phi.get(1, 2, 2) == approx(13.0));

  check_tensor_doubles_equals_tensor_datavectors(
      GeneralizedHarmonic::phi(
          make_lapse(used_for_size), make_deriv_lapse<dim>(used_for_size),
          make_shift<dim>(used_for_size), make_deriv_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_deriv_spatial_metric<dim>(used_for_size)),
      phi);
}

void test_compute_3d_phi(const DataVector& used_for_size) {
  const size_t dim = 3;
  const auto phi = GeneralizedHarmonic::phi(
      make_lapse(0.), make_deriv_lapse<dim>(0.), make_shift<dim>(0.),
      make_deriv_shift<dim>(0.), make_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(phi.get(0, 0, 0) == approx(2421.0));
  CHECK(phi.get(0, 0, 1) == approx(108.0));
  CHECK(phi.get(0, 0, 2) == approx(216.0));
  CHECK(phi.get(0, 0, 3) == approx(324.0));
  CHECK(phi.get(0, 1, 1) == approx(3.0));
  CHECK(phi.get(0, 1, 2) == approx(6.0));
  CHECK(phi.get(0, 1, 3) == approx(9.0));
  CHECK(phi.get(0, 2, 2) == approx(12.0));
  CHECK(phi.get(0, 2, 3) == approx(18.0));
  CHECK(phi.get(0, 3, 3) == approx(27.0));
  CHECK(phi.get(1, 0, 0) == approx(2274.0));
  CHECK(phi.get(1, 0, 1) == approx(108.0));
  CHECK(phi.get(1, 0, 2) == approx(210.0));
  CHECK(phi.get(1, 0, 3) == approx(312.0));
  CHECK(phi.get(1, 1, 1) == approx(4.0));
  CHECK(phi.get(1, 1, 2) == approx(7.0));
  CHECK(phi.get(1, 1, 3) == approx(10.0));
  CHECK(phi.get(1, 2, 2) == approx(13.0));
  CHECK(phi.get(1, 2, 3) == approx(19.0));
  CHECK(phi.get(1, 3, 3) == approx(28.0));
  CHECK(phi.get(2, 0, 0) == approx(2127.0));
  CHECK(phi.get(2, 0, 1) == approx(108.0));
  CHECK(phi.get(2, 0, 2) == approx(204.0));
  CHECK(phi.get(2, 0, 3) == approx(300.0));
  CHECK(phi.get(2, 1, 1) == approx(5.0));
  CHECK(phi.get(2, 1, 2) == approx(8.0));
  CHECK(phi.get(2, 1, 3) == approx(11.0));
  CHECK(phi.get(2, 2, 2) == approx(14.0));
  CHECK(phi.get(2, 2, 3) == approx(20.0));
  CHECK(phi.get(2, 3, 3) == approx(29.0));

  check_tensor_doubles_equals_tensor_datavectors(
      GeneralizedHarmonic::phi(
          make_lapse(used_for_size), make_deriv_lapse<dim>(used_for_size),
          make_shift<dim>(used_for_size), make_deriv_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_deriv_spatial_metric<dim>(used_for_size)),
      phi);
}

void test_compute_1d_pi(const DataVector& used_for_size) {
  const size_t dim = 1;
  const auto pi = GeneralizedHarmonic::pi(
      make_lapse(0.), make_dt_lapse(0.), make_shift<dim>(0.),
      make_dt_shift<dim>(0.), make_spatial_metric<dim>(0.),
      make_dt_spatial_metric<dim>(0.),
      make_spatial_deriv_spacetime_metric<dim>(0.));

  CHECK(pi.get(0, 0) == approx(31. / 3.));
  CHECK(pi.get(0, 1) == approx(5. / 3.));
  CHECK(pi.get(1, 1) == approx(4.0));

  check_tensor_doubles_equals_tensor_datavectors(
      GeneralizedHarmonic::pi(
          make_lapse(used_for_size), make_dt_lapse(used_for_size),
          make_shift<dim>(used_for_size), make_dt_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_dt_spatial_metric<dim>(used_for_size),
          make_spatial_deriv_spacetime_metric<dim>(used_for_size)),
      pi);
}

void test_compute_2d_pi(const DataVector& used_for_size) {
  const size_t dim = 2;
  const auto pi = GeneralizedHarmonic::pi(
      make_lapse(0.), make_dt_lapse(0.), make_shift<dim>(0.),
      make_dt_shift<dim>(0.), make_spatial_metric<dim>(0.),
      make_dt_spatial_metric<dim>(0.),
      make_spatial_deriv_spacetime_metric<dim>(0.));

  CHECK(pi.get(0, 0) == approx(-71. / 3.));
  CHECK(pi.get(0, 1) == approx(10. / 3.));
  CHECK(pi.get(0, 2) == approx(8. / 3.));
  CHECK(pi.get(1, 1) == approx(44. / 3.));
  CHECK(pi.get(1, 2) == approx(65. / 3.));
  CHECK(pi.get(2, 2) == approx(97. / 3.));

  check_tensor_doubles_equals_tensor_datavectors(
      GeneralizedHarmonic::pi(
          make_lapse(used_for_size), make_dt_lapse(used_for_size),
          make_shift<dim>(used_for_size), make_dt_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_dt_spatial_metric<dim>(used_for_size),
          make_spatial_deriv_spacetime_metric<dim>(used_for_size)),
      pi);
}

void test_compute_3d_pi(const DataVector& used_for_size) {
  const size_t dim = 3;
  const auto pi = GeneralizedHarmonic::pi(
      make_lapse(0.), make_dt_lapse(0.), make_shift<dim>(0.),
      make_dt_shift<dim>(0.), make_spatial_metric<dim>(0.),
      make_dt_spatial_metric<dim>(0.),
      make_spatial_deriv_spacetime_metric<dim>(0.));

  CHECK(pi.get(0, 0) == approx(-1216. / 3.));
  CHECK(pi.get(0, 1) == approx(2. / 3.));
  CHECK(pi.get(0, 2) == approx(-20. / 3.));
  CHECK(pi.get(0, 3) == approx(-14.0));
  CHECK(pi.get(1, 1) == approx(104. / 3.));
  CHECK(pi.get(1, 2) == approx(155. / 3.));
  CHECK(pi.get(1, 3) == approx(206. / 3.));
  CHECK(pi.get(2, 2) == approx(232. / 3.));
  CHECK(pi.get(2, 3) == approx(103.0));
  CHECK(pi.get(3, 3) == approx(412. / 3.));

  check_tensor_doubles_equals_tensor_datavectors(
      GeneralizedHarmonic::pi(
          make_lapse(used_for_size), make_dt_lapse(used_for_size),
          make_shift<dim>(used_for_size), make_dt_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_dt_spatial_metric<dim>(used_for_size),
          make_spatial_deriv_spacetime_metric<dim>(used_for_size)),
      pi);
}

void test_compute_1d_gauge_source(const DataVector& used_for_size) {
  const size_t dim = 1;
  const auto gauge_source = GeneralizedHarmonic::gauge_source(
      make_lapse(0.), make_dt_lapse(0.), make_deriv_lapse<dim>(0.),
      make_shift<dim>(0.), make_dt_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_trace_extrinsic_curvature(0.),
      make_trace_spatial_christoffel_first_kind<dim>(0.));

  CHECK(gauge_source.get(0) == approx(-41. / 3.));
  CHECK(gauge_source.get(1) == approx(13. / 6.));

  check_tensor_doubles_approx_equals_tensor_datavectors(
      GeneralizedHarmonic::gauge_source(
          make_lapse(used_for_size), make_dt_lapse(used_for_size),
          make_deriv_lapse<dim>(used_for_size), make_shift<dim>(used_for_size),
          make_dt_shift<dim>(used_for_size),
          make_deriv_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_trace_extrinsic_curvature(used_for_size),
          make_trace_spatial_christoffel_first_kind<dim>(used_for_size)),
      gauge_source);
}

void test_compute_2d_gauge_source(const DataVector& used_for_size) {
  const size_t dim = 2;
  const auto gauge_source = GeneralizedHarmonic::gauge_source(
      make_lapse(0.), make_dt_lapse(0.), make_deriv_lapse<dim>(0.),
      make_shift<dim>(0.), make_dt_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_trace_extrinsic_curvature(0.),
      make_trace_spatial_christoffel_first_kind<dim>(0.));

  CHECK(gauge_source.get(0) == approx(-400. / 9.));
  CHECK(gauge_source.get(1) == approx(-395. / 90.));
  CHECK(gauge_source.get(2) == approx(-124. / 9.));

  check_tensor_doubles_approx_equals_tensor_datavectors(
      GeneralizedHarmonic::gauge_source(
          make_lapse(used_for_size), make_dt_lapse(used_for_size),
          make_deriv_lapse<dim>(used_for_size), make_shift<dim>(used_for_size),
          make_dt_shift<dim>(used_for_size),
          make_deriv_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_trace_extrinsic_curvature(used_for_size),
          make_trace_spatial_christoffel_first_kind<dim>(used_for_size)),
      gauge_source);
}

void test_compute_3d_gauge_source(const DataVector& used_for_size) {
  const size_t dim = 3;
  const auto gauge_source = GeneralizedHarmonic::gauge_source(
      make_lapse(0.), make_dt_lapse(0.), make_deriv_lapse<dim>(0.),
      make_shift<dim>(0.), make_dt_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_trace_extrinsic_curvature(0.),
      make_trace_spatial_christoffel_first_kind<dim>(0.));

  CHECK(gauge_source.get(0) == approx(-1444. / 3.));
  CHECK(gauge_source.get(1) == approx(-187. / 6.));
  CHECK(gauge_source.get(2) == approx(-202. / 3.));
  CHECK(gauge_source.get(3) == approx(-207. / 2.));

  check_tensor_doubles_approx_equals_tensor_datavectors(
      GeneralizedHarmonic::gauge_source(
          make_lapse(used_for_size), make_dt_lapse(used_for_size),
          make_deriv_lapse<dim>(used_for_size), make_shift<dim>(used_for_size),
          make_dt_shift<dim>(used_for_size),
          make_deriv_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_trace_extrinsic_curvature(used_for_size),
          make_trace_spatial_christoffel_first_kind<dim>(used_for_size)),
      gauge_source);
}

template <size_t Dim, typename T>
void test_compute_extrinsic_curvature_and_deriv_metric(const T& used_for_size) {
  // Set up random values for lapse, shift, spatial_metric,
  // and their derivatives.
  std::random_device r;
  const auto seed = r();
  std::mt19937 generator(seed);
  INFO("seed" << seed);
  std::uniform_real_distribution<> dist(-1., 1.);
  std::uniform_real_distribution<> dist_positive(1., 2.);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);
  const auto nn_dist_positive = make_not_null(&dist_positive);

  const auto lapse = make_with_random_values<Scalar<T>>(
      nn_generator, nn_dist_positive, used_for_size);
  const auto shift = make_with_random_values<tnsr::I<T, Dim>>(
      nn_generator, nn_dist, used_for_size);
  const auto spatial_metric = [&]() {
    auto spatial_metric_l = make_with_random_values<tnsr::ii<T, Dim>>(
        nn_generator, nn_dist, used_for_size);
    // Make sure spatial_metric isn't singular by adding
    // large enough positive diagonal values.
    for (size_t i = 0; i < Dim; ++i) {
      spatial_metric_l.get(i, i) += 4.0;
    }
    return spatial_metric_l;
  }();
  const auto dt_lapse = make_with_random_values<Scalar<T>>(
      nn_generator, nn_dist_positive, used_for_size);
  const auto deriv_lapse = make_with_random_values<tnsr::i<T, Dim>>(
      nn_generator, nn_dist_positive, used_for_size);
  const auto dt_shift = make_with_random_values<tnsr::I<T, Dim>>(
      nn_generator, nn_dist, used_for_size);
  const auto deriv_shift = make_with_random_values<tnsr::iJ<T, Dim>>(
      nn_generator, nn_dist, used_for_size);
  const auto deriv_spatial_metric = make_with_random_values<tnsr::ijj<T, Dim>>(
      nn_generator, nn_dist, used_for_size);
  const auto dt_spatial_metric = make_with_random_values<tnsr::ii<T, Dim>>(
      nn_generator, nn_dist, used_for_size);

  // Make extrinsic curvature, spacetime_normal_vector, and generalized
  // harmonic pi,psi variables in a way that is already independently tested.
  const auto extrinsic_curvature =
      gr::extrinsic_curvature(lapse, shift, deriv_shift, spatial_metric,
                              dt_spatial_metric, deriv_spatial_metric);
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);
  const auto phi =
      GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                               spatial_metric, deriv_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);

  // Compute extrinsic curvature and deriv_spatial_metric from generalized
  // harmonic variables and make sure we get the same result.
  const auto extrinsic_curvature_test =
      GeneralizedHarmonic::extrinsic_curvature(spacetime_normal_vector, pi,
                                               phi);
  const auto deriv_spatial_metric_test =
      GeneralizedHarmonic::deriv_spatial_metric(phi);

  CHECK_ITERABLE_APPROX(extrinsic_curvature, extrinsic_curvature_test);
  CHECK_ITERABLE_APPROX(deriv_spatial_metric, deriv_spatial_metric_test);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.GhQuantities",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv(2);
  test_compute_1d_phi(dv);
  test_compute_2d_phi(dv);
  test_compute_3d_phi(dv);
  test_compute_1d_pi(dv);
  test_compute_2d_pi(dv);
  test_compute_3d_pi(dv);
  test_compute_1d_gauge_source(dv);
  test_compute_2d_gauge_source(dv);
  test_compute_3d_gauge_source(dv);
  test_compute_extrinsic_curvature_and_deriv_metric<1>(0.);
  test_compute_extrinsic_curvature_and_deriv_metric<2>(0.);
  test_compute_extrinsic_curvature_and_deriv_metric<3>(0.);
  test_compute_extrinsic_curvature_and_deriv_metric<1>(dv);
  test_compute_extrinsic_curvature_and_deriv_metric<2>(dv);
  test_compute_extrinsic_curvature_and_deriv_metric<3>(dv);
}
