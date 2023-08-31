// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/StrahlkorperTestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
void test_normals() {
  const double y11_amplitude = 1.0;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const auto strahlkorper =
      ylm::TestHelpers::create_strahlkorper_y11(y11_amplitude, radius, center);

  const auto theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  const auto n_pts = theta_phi[0].size();

  const double amp = -sqrt(3.0 / 8.0 / M_PI) * y11_amplitude;
  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];
  const DataVector cos_phi = cos(phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_theta = cos(theta);
  const DataVector sin_theta = sin(theta);

  auto box = db::create<
      db::AddSimpleTags<ylm::Tags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<ylm::Tags::compute_items_tags<Frame::Inertial>>>(
      strahlkorper);

  // Test surface_normal_magnitude.
  tnsr::II<DataVector, 3, Frame::Inertial> invg(n_pts);
  invg.get(0, 0) = 1.0;
  invg.get(1, 0) = 0.1;
  invg.get(2, 0) = 0.2;
  invg.get(1, 1) = 2.0;
  invg.get(1, 2) = 0.3;
  invg.get(2, 2) = 3.0;

  const auto expected_normal_mag = [&]() -> DataVector {
    const auto& r = get(db::get<ylm::Tags::Radius<Frame::Inertial>>(box));

    // Nasty expression Mark Scheel computed in Mathematica.
    const DataVector normsquared =
        (-0.3 * cos_theta * (r + amp * sin_phi * sin_theta) *
             (1. * amp * square(cos_phi) +
              1. * amp * square(cos_theta) * square(sin_phi) -
              1. * r * sin_phi * sin_theta +
              cos_phi * (2.0 / 3.0) *
                  (amp * (-1. + square(cos_theta)) * sin_phi - r * sin_theta) +
              cos_theta * (-10. * r - 10. * amp * sin_phi * sin_theta)) +
         0.1 * cos_phi *
             (amp * (-1. + 1. * square(cos_theta)) * sin_phi -
              1. * r * sin_theta) *
             (1. * amp * square(cos_phi) +
              1. * amp * square(cos_theta) * square(sin_phi) -
              1. * r * sin_phi * sin_theta +
              cos_phi * (amp * (-10. + 10. * square(cos_theta)) * sin_phi -
                         10. * r * sin_theta) +
              cos_theta * (-2. * r - 2. * amp * sin_phi * sin_theta)) +
         2. *
             (amp * square(cos_phi) +
              sin_phi *
                  (amp * square(cos_theta) * sin_phi - 1. * r * sin_theta)) *
             (amp * square(cos_phi) +
              amp * square(cos_theta) * square(sin_phi) -
              1. * r * sin_phi * sin_theta +
              cos_phi * 0.05 *
                  (amp * (-1. + square(cos_theta)) * sin_phi - r * sin_theta) +
              cos_theta * (-0.15 * r - 0.15 * amp * sin_phi * sin_theta))) /
        square(r);
    return sqrt(normsquared);
  }();
  const auto& normal_one_form =
      db::get<ylm::Tags::NormalOneForm<Frame::Inertial>>(box);
  const auto normal_mag = magnitude(normal_one_form, invg);
  CHECK_ITERABLE_APPROX(expected_normal_mag, get(normal_mag));
}

void test_max_ricci_scalar() {
  // Test max_ricci_scalar
  Scalar<DataVector> d{{{{1., 2., 3.}}}};
  const double expected_max{3.};
  double max{std::numeric_limits<double>::signaling_NaN()};
  ylm::Tags::MaxRicciScalarCompute::function(make_not_null(&max), d);
  CHECK(expected_max == max);
}

void test_min_ricci_scalar() {
  // Test min_ricci_scalar
  Scalar<DataVector> d{{{{1., 2., 3.}}}};
  const double expected_min{1.};
  double min{std::numeric_limits<double>::signaling_NaN()};
  ylm::Tags::MinRicciScalarCompute::function(make_not_null(&min), d);
  CHECK(expected_min == min);
}

void test_dimensionful_spin_vector_compute_tag() {
  const double y11_amplitude = 1.0;
  const double y11_radius = 2.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const auto strahlkorper = ylm::TestHelpers::create_strahlkorper_y11(
      y11_amplitude, y11_radius, center);
  const size_t ylm_physical_size =
      strahlkorper.ylm_spherepack().physical_size();
  const DataVector used_for_size(ylm_physical_size,
                                 std::numeric_limits<double>::signaling_NaN());

  // Creates a variable named generator that can be used to generate random
  // values
  MAKE_GENERATOR(generator);
  // Creates a uniform distribution, which will be used to generate random
  // numbers
  std::uniform_real_distribution<> dist(-1., 1.);

  const double dimensionful_spin_magnitude{5.0};
  // Create the tensor arguments to spin_vector by having them contain
  // DataVectors of size == ylm physical size, where values are random
  const auto area_element = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), dist, used_for_size);
  const auto radius = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), dist, used_for_size);
  const auto r_hat =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), dist, used_for_size);
  const auto ricci_scalar = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), dist, used_for_size);
  const auto spin_function = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), dist, used_for_size);
  const auto strahlkorper_cartesian_coords =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), dist, used_for_size);
  const auto box = db::create<
      db::AddSimpleTags<
          gr::surfaces::Tags::DimensionfulSpinMagnitude,
          gr::surfaces::Tags::AreaElement<Frame::Inertial>,
          ylm::Tags::Radius<Frame::Inertial>, ylm::Tags::Rhat<Frame::Inertial>,
          ylm::Tags::RicciScalar, gr::surfaces::Tags::SpinFunction,
          ylm::Tags::Strahlkorper<Frame::Inertial>,
          ylm::Tags::CartesianCoords<Frame::Inertial>>,
      db::AddComputeTags<gr::surfaces::Tags::DimensionfulSpinVectorCompute<
          Frame::Inertial, Frame::Inertial>>>(
      dimensionful_spin_magnitude, area_element, radius, r_hat, ricci_scalar,
      spin_function, strahlkorper, strahlkorper_cartesian_coords);
  // LHS of the == in the CHECK is retrieving the computed dimensionful spin
  // vector from your DimensionfulSpinVectorCompute tag and RHS of ==
  // should be same logic as DimensionfulSpinVectorCompute::function
  CHECK(db::get<gr::surfaces::Tags::DimensionfulSpinVector<Frame::Inertial>>(
            box) ==
        gr::surfaces::spin_vector<Frame::Inertial, Frame::Inertial>(
            dimensionful_spin_magnitude, area_element, ricci_scalar,
            spin_function, strahlkorper, strahlkorper_cartesian_coords));
}

void test_dimensionless_spin_magnitude_compute_tag() {
  const double dimensionful_spin_magnitude{5.0};
  const double christodoulou_mass = 4.444;

  const auto box = db::create<
      db::AddSimpleTags<gr::surfaces::Tags::DimensionfulSpinMagnitude,
                        gr::surfaces::Tags::ChristodoulouMass>,
      db::AddComputeTags<gr::surfaces::Tags::DimensionlessSpinMagnitudeCompute<
          Frame::Inertial>>>(dimensionful_spin_magnitude, christodoulou_mass);
  // LHS of the == in the CHECK is retrieving the computed dimensionless spin
  // magnitude from your DimensionlessSpinMagnitudeCompute tag and RHS of ==
  // should be same logic as DimensionlessSpinMagnitudeCompute::function
  CHECK(
      db::get<gr::surfaces::Tags::DimensionlessSpinMagnitude<Frame::Inertial>>(
          box) ==
      gr::surfaces::dimensionless_spin_magnitude(dimensionful_spin_magnitude,
                                                 christodoulou_mass));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperDataBox",
                  "[ApparentHorizons][Unit]") {
  test_normals();
  test_max_ricci_scalar();
  test_min_ricci_scalar();
  test_dimensionful_spin_vector_compute_tag();
  test_dimensionless_spin_magnitude_compute_tag();
  TestHelpers::db::test_simple_tag<ah::Tags::FastFlow>("FastFlow");
  TestHelpers::db::test_base_tag<ah::Tags::ObserveCentersBase>(
      "ObserveCentersBase");
  TestHelpers::db::test_simple_tag<ah::Tags::ObserveCenters>("ObserveCenters");
}
