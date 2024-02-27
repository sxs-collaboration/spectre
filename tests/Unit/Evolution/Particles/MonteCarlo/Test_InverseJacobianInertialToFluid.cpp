// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Particles/MonteCarlo/InverseJacobianInertialToFluidCompute.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Particles.MonteCarlo.InverseJacobianInertialToFluid",
    "[Unit][Evolution]") {
  const DataVector used_for_size(5);
  MAKE_GENERATOR(generator);

  const double epsilon_approx = 5.e-12;

  const auto lapse =
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size);
  const auto shift = TestHelpers::gr::random_shift<3>(make_not_null(&generator),
                                                      used_for_size);
  const auto spatial_metric = TestHelpers::gr::random_spatial_metric<3>(
      make_not_null(&generator), used_for_size);
  const auto lorentz_factor = TestHelpers::hydro::random_lorentz_factor(
      make_not_null(&generator), used_for_size);
  const auto spatial_velocity =
      TestHelpers::hydro::random_velocity<DataVector, 3>(
          make_not_null(&generator), lorentz_factor, spatial_metric);

  TestHelpers::db::test_compute_tag<
      Particles::MonteCarlo::InverseJacobianInertialToFluidCompute>(
      "InverseJacobian(Inertial,Fluid)");
  const auto box = db::create<
      db::AddSimpleTags<
          hydro::Tags::LorentzFactor<DataVector>,
          hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
          gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
          gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>,
      db::AddComputeTags<
          Particles::MonteCarlo::InverseJacobianInertialToFluidCompute>>(
      lorentz_factor, spatial_velocity, lapse, shift, spatial_metric);

  const auto& inverse_jacobian =
      db::get<Particles::MonteCarlo::InverseJacobianInertialToFluidCompute>(
          box);
  const tnsr::aa<DataVector, 3, Frame::Inertial> spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);

  // Check that the time vector is u^mu
  CHECK_ITERABLE_CUSTOM_APPROX(
      inverse_jacobian.get(0, 0), get(lorentz_factor) / get(lapse),
      Approx::custom().epsilon(epsilon_approx).scale(1.0));
  for (size_t d = 0; d < 3; d++) {
    CHECK_ITERABLE_CUSTOM_APPROX(
        inverse_jacobian.get(d + 1, 0),
        get(lorentz_factor) *
            (spatial_velocity.get(d) - shift.get(d) / get(lapse)),
        Approx::custom().epsilon(epsilon_approx).scale(1.0));
  }

  // Test that we have orthonormal tetrads
  DataVector dot_product(used_for_size);
  DataVector expected_dot_product(used_for_size);
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = a; b < 4; b++) {
      dot_product = 0.0;
      for (size_t d = 0; d < 4; d++) {
        for (size_t dd = 0; dd < 4; dd++) {
          dot_product += spacetime_metric.get(d, dd) *
                         inverse_jacobian.get(d, a) *
                         inverse_jacobian.get(dd, b);
        }
      }
      expected_dot_product = (a == b ? (a == 0 ? -1.0 : 1.0) : 0.0);
      CHECK_ITERABLE_CUSTOM_APPROX(
          dot_product, expected_dot_product,
          Approx::custom().epsilon(epsilon_approx).scale(1.0));
    }
  }
}
