// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/CubicCurvatureScalars.hpp"
#include "PointwiseFunctions/GeneralRelativity/QuadraticCurvatureScalars.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

template <typename DataType>
void build_schwarzschild_spatial_metrics(
    gsl::not_null<tnsr::ii<DataType, 3, Frame::Inertial>*> spatial_metric,
    gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
        inverse_spatial_metric,
    const DataType& r, const DataType& theta, const double mass) {
  const auto one = make_with_value<DataType>(r, 1.0);
  const auto sin_th = sin(theta);
  const auto g_rr = one / (one - 2.0 * mass / r);
  const auto g_tt = r * r;
  const auto g_pp = g_tt * sin_th * sin_th;

  spatial_metric->get(0, 0) = g_rr;
  spatial_metric->get(1, 1) = g_tt;
  spatial_metric->get(2, 2) = g_pp;

  inverse_spatial_metric->get(0, 0) = one - 2.0 * mass / r;
  inverse_spatial_metric->get(1, 1) = one / (r * r);
  inverse_spatial_metric->get(2, 2) = one / g_pp;
}

void test_curvature_scalars_schwarzschild() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> mass_dist(0.5, 2.0);
  std::uniform_real_distribution<> rfac_dist(2.5, 6.0);
  std::uniform_real_distribution<> theta_dist(0.3, 2.5);
  const double mass = mass_dist(gen);
  const size_t num_points = 1000;
  const auto r = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&rfac_dist), num_points);
  const auto theta = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&theta_dist), num_points);
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric(num_points, 0.0);
  tnsr::II<DataVector, 3, Frame::Inertial> inverse_spatial_metric(num_points,
                                                                  0.0);
  build_schwarzschild_spatial_metrics(make_not_null(&spatial_metric),
                                      make_not_null(&inverse_spatial_metric), r,
                                      theta, mass);

  const DataVector a = mass / (r * r * r);
  tnsr::ii<DataVector, 3, Frame::Inertial> electric_weyl(num_points, 0.0);
  const tnsr::ii<DataVector, 3, Frame::Inertial> magnetic_weyl(num_points, 0.0);
  electric_weyl.get(0, 0) = -2.0 * a * spatial_metric.get(0, 0);
  electric_weyl.get(1, 1) = a * spatial_metric.get(1, 1);
  electric_weyl.get(2, 2) = a * spatial_metric.get(2, 2);

  const auto electric_weyl_scalar =
      gr::weyl_electric_scalar(electric_weyl, inverse_spatial_metric);
  const auto magnetic_weyl_scalar =
      gr::weyl_magnetic_scalar<Frame::Inertial, DataVector>(
          magnetic_weyl, inverse_spatial_metric);

  const auto kretschmann = gr::kretschmann_scalar_in_vacuum<DataVector>(
      electric_weyl_scalar, magnetic_weyl_scalar);
  CHECK_ITERABLE_APPROX(kretschmann.get(), 48.0 * a * a);

  const auto gauss_bonnet = gr::gauss_bonnet_scalar_in_vacuum<DataVector>(
      electric_weyl_scalar, magnetic_weyl_scalar);
  CHECK_ITERABLE_APPROX(gauss_bonnet.get(), 48.0 * a * a);

  const auto pontryagin = gr::pontryagin_scalar<DataVector, Frame::Inertial>(
      electric_weyl, magnetic_weyl, inverse_spatial_metric);
  CHECK_ITERABLE_APPROX(pontryagin.get(), DataVector(num_points, 0.0));

  const auto cubic_invariant_real = gr::cubic_invariant_real(
      electric_weyl, magnetic_weyl, inverse_spatial_metric);
  const auto cubic_invariant_imag = gr::cubic_invariant_imag(
      electric_weyl, magnetic_weyl, inverse_spatial_metric);
  CHECK_ITERABLE_APPROX(cubic_invariant_real.get(), a * a * a);
  CHECK_ITERABLE_APPROX(cubic_invariant_imag.get(),
                        DataVector(num_points, 0.0));
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.CurvatureScalars.Schwarzschild",
    "[PointwiseFunctions][Unit]") {
  test_curvature_scalars_schwarzschild();
}
