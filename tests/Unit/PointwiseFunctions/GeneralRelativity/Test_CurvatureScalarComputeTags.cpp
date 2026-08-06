// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/CubicCurvatureScalars.hpp"
#include "PointwiseFunctions/GeneralRelativity/QuadraticCurvatureScalars.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"
#include "Utilities/Gsl.hpp"

namespace {
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.CurvatureScalars.ComputeTags",
    "[PointwiseFunctions][Unit]") {
  TestHelpers::db::test_compute_tag<
      gr::Tags::PontryaginScalarCompute<DataVector, Frame::Inertial>>(
      "PontryaginScalar");
  TestHelpers::db::test_compute_tag<
      gr::Tags::KretschmannScalarCompute<DataVector>>("KretschmannScalar");
  TestHelpers::db::test_compute_tag<
      gr::Tags::GaussBonnetScalarCompute<DataVector>>("GaussBonnetScalar");
  TestHelpers::db::test_compute_tag<
      gr::Tags::CubicInvariantRealCompute<DataVector, 3, Frame::Inertial>>(
      "CubicInvariantReal");
  TestHelpers::db::test_compute_tag<
      gr::Tags::CubicInvariantImagCompute<DataVector, 3, Frame::Inertial>>(
      "CubicInvariantImag");

  const DataVector used_for_size(7);
  // Randomize inputs inline
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-0.2, 0.2);
  const auto E =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto B =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  auto inv_metric =
      make_with_random_values<tnsr::II<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  for (size_t i = 0; i < 3; ++i) {
    inv_metric.get(i, i) += 1.0;
  }

  const auto E_scalar = gr::weyl_electric_scalar(E, inv_metric);
  const auto B_scalar =
      gr::weyl_magnetic_scalar<Frame::Inertial, DataVector>(B, inv_metric);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::WeylElectric<DataVector, 3, Frame::Inertial>,
          gr::Tags::WeylMagnetic<DataVector, 3, Frame::Inertial>,
          gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
          gr::Tags::WeylElectricScalar<DataVector>,
          gr::Tags::WeylMagneticScalar<DataVector>>,
      db::AddComputeTags<
          gr::Tags::PontryaginScalarCompute<DataVector, Frame::Inertial>,
          gr::Tags::GaussBonnetScalarCompute<DataVector>,
          gr::Tags::KretschmannScalarCompute<DataVector>,
          gr::Tags::CubicInvariantRealCompute<DataVector, 3, Frame::Inertial>,
          gr::Tags::CubicInvariantImagCompute<DataVector, 3, Frame::Inertial>>>(
      E, B, inv_metric, E_scalar, B_scalar);

  const auto expected_pontryagin =
      gr::pontryagin_scalar<DataVector, Frame::Inertial>(E, B, inv_metric);
  const auto expected_kretschmann =
      gr::kretschmann_scalar_in_vacuum<DataVector>(E_scalar, B_scalar);
  const auto expected_gb =
      gr::gauss_bonnet_scalar_in_vacuum<DataVector>(E_scalar, B_scalar);
  const auto expected_cubic_real = gr::cubic_invariant_real(E, B, inv_metric);
  const auto expected_cubic_imag = gr::cubic_invariant_imag(E, B, inv_metric);

  CHECK_ITERABLE_APPROX((db::get<gr::Tags::PontryaginScalar<DataVector>>(box)),
                        expected_pontryagin);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::KretschmannScalar<DataVector>>(box)),
                        expected_kretschmann);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::GaussBonnetScalar<DataVector>>(box)),
                        expected_gb);
  CHECK_ITERABLE_APPROX(
      (db::get<gr::Tags::CubicInvariantReal<DataVector>>(box)),
      expected_cubic_real);
  CHECK_ITERABLE_APPROX(
      (db::get<gr::Tags::CubicInvariantImag<DataVector>>(box)),
      expected_cubic_imag);
}

}  // namespace
