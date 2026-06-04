// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/Hydro/Ricci.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"
#include "PointwiseFunctions/Hydro/WeylElectric.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

void test_curvature_quantities_flrw() {
  // The FLRW metric provides a non-trivial but still analytic spacetime that
  // depends on a nonzero stress-energy tensor, allowing for a simple test of
  // the various curvature tensors/scalars (Ricci and Kretschmann) in a hydro
  // environment.
  //
  // We assume zero spatial curvature and zero cosmological constant. Then, the
  // line element is given by $ds^2 = -dt^2 + a(t)^2 d\|\vec{x}\|^2$, which
  // yields:
  // - A diagonal Ricci tensor: $R_{tt} = -3 \ddot{a} / a$, $R_{ii} = (a\ddot{a}
  //   + 2\dot{a}^2)$
  // - Ricci scalar: $R = 6 (\ddot{a}/a + (\dot{a} / a)^2 )$
  // - Kretschmann scalar: $K = (6 / a^8) (2\dot{a}^4 + a^2 (1 + a^4)
  //   \ddot{a}^2)$
  //
  // The stress-energy tensor is given by the usual cold ideal fluid expression:
  // $T^{ab} = (\rho + p) u^a u^b + p g^{ab}$, where $u^a = (1, 0, 0, 0)$ is a
  // stationary 4-velocity, $\rho$ is the rest mass density, and $p$ the
  // pressure. Applying the Einstein Equations yields the Friedmann Equations,
  // which allows us to relate the density and pressure to the scale factor and
  // its derivatives:
  // $(\dot{a} / a)^2 = 8\pi \rho / 3$
  // $\ddot{a} / a = - (4\pi / 3) (\rho + 3p)$
  // Substituting these into the above yields the following curvature
  // quantities:
  // - Ricci tensor: $R_{tt} = 4\pi (3p + \rho)$, $R_{ii} = 4\pi a^2 (\rho - p)$
  // - Ricci scalar: $R = 8\pi (\rho - 3p)$
  // - Kretschmann scalar: $K = (64\pi^2 / 3) * (4\rho^2 + (3p + \rho)^2)$
  //
  // The metric is constant in space. We test random values for density,
  // pressure, and the scale factor, to represent random instances of the
  // metric, not random points.

  // Random parameters for metric and Friedmann Equations
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> density_dist(1.0, 2.0);
  std::uniform_real_distribution<> pressure_dist(0.0, 0.3);
  std::uniform_real_distribution<> scale_factor_dist(0.3, 2.5);
  const size_t num_points = 100;
  Scalar<DataVector> density(num_points, 0.0);
  get(density) = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&density_dist), num_points);
  Scalar<DataVector> pressure(num_points, 0.0);
  get(pressure) = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&pressure_dist), num_points);
  const auto scale_factor = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&scale_factor_dist), num_points);

  // Compute analytic expressions for metric and its derivatives
  auto compute_spacetime_metric = [num_points, scale_factor]() {
    // $ds^2 = -dt^2 + a(t)^2 d\|\vec{x}\|^2$
    tnsr::aa<DataVector, 3, Frame::Inertial> spacetime_metric(num_points, 0.0);
    spacetime_metric.get(0, 0) = -1.;
    for (size_t i = 0; i < 3; ++i) {
      spacetime_metric.get(i + 1, i + 1) = square(scale_factor);
    }
    return spacetime_metric;
  };
  const tnsr::aa<DataVector, 3, Frame::Inertial> spacetime_metric =
      compute_spacetime_metric();
  const tnsr::AA<DataVector, 3, Frame::Inertial> inverse_spacetime_metric =
      determinant_and_inverse(spacetime_metric).second;

  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const tnsr::II<DataVector, 3, Frame::Inertial> inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const tnsr::I<DataVector, 3, Frame::Inertial> shift(num_points, 0.0);
  const Scalar<DataVector> lapse(num_points, 1.0);

  // Hydro quantities for stress energy tensor (cold stationary fluid)
  const Scalar<DataVector> specific_internal_energy(num_points, 0.0);
  const Scalar<DataVector> lorentz_factor(num_points, 1.0);
  const Scalar<DataVector> comoving_magnetic_field_magnitude(num_points, 0.0);
  const tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity(num_points,
                                                                 0.0);
  const tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field(num_points, 0.0);

  // Stress energy tensor
  tnsr::AA<DataVector, 3, Frame::Inertial> stress_energy(num_points, 0.0);
  hydro::stress_energy_tensor(
      make_not_null(&stress_energy), density, specific_internal_energy,
      pressure, lorentz_factor, lapse, comoving_magnetic_field_magnitude,
      spatial_velocity, shift, magnetic_field, spatial_metric,
      inverse_spatial_metric);

  // Ricci computation and tests
  const auto ricci = hydro::ricci_in_gr(stress_energy, spacetime_metric);
  CHECK_ITERABLE_APPROX(ricci.get(0, 0),
                        4. * M_PI * (get(density) + 3. * get(pressure)));
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_APPROX(
        ricci.get(i + 1, i + 1),
        4. * M_PI * square(scale_factor) * (get(density) - get(pressure)));
  }
  const auto ricci_scalar = gr::ricci_scalar(ricci, inverse_spacetime_metric);
  CHECK_ITERABLE_APPROX(get(ricci_scalar),
                        8. * M_PI * (get(density) - 3. * get(pressure)));

  auto compute_deriv_spacetime_metric = [num_points, scale_factor, density]() {
    // Friedmann equations ->
    // $d_t g_{ii} = 2 a \dot{a} = 2 a^2 \sqrt{8\pi \rho / 3}
    // All other components are 0.
    tnsr::abb<DataVector, 3, Frame::Inertial> deriv_spacetime_metric(num_points,
                                                                     0.0);
    tnsr::ii<DataVector, 3, Frame::Inertial> dt_spatial_metric(num_points, 0.0);
    for (size_t i = 0; i < 3; ++i) {
      deriv_spacetime_metric.get(0, i + 1, i + 1) =
          2. * square(scale_factor) * sqrt(8. * M_PI * get(density) / 3.);
      dt_spatial_metric.get(i, i) =
          2. * square(scale_factor) * sqrt(8. * M_PI * get(density) / 3.);
    }
    return std::make_pair(deriv_spacetime_metric, dt_spatial_metric);
  };
  const auto derivs = compute_deriv_spacetime_metric();
  const tnsr::ii<DataVector, 3, Frame::Inertial> dt_spatial_metric =
      derivs.second;
  const tnsr::ijj<DataVector, 3, Frame::Inertial> deriv_spatial_metric(
      num_points, 0.0);
  const tnsr::iJ<DataVector, 3, Frame::Inertial> deriv_shift(num_points, 0.0);
  const auto induced_spatial_metric =
      gr::induced_spatial_metric(spacetime_metric, lapse);
  const tnsr::ii<DataVector, 3, Frame::Inertial> spatial_ricci(num_points, 0.0);

  const auto extrinsic_curvature =
      gr::extrinsic_curvature(lapse, shift, deriv_shift, spatial_metric,
                              dt_spatial_metric, deriv_spatial_metric);

  // Weyl electric and magnetic components
  const auto electric_weyl =
      hydro::weyl_electric(gr::weyl_electric(spatial_ricci, extrinsic_curvature,
                                             inverse_spatial_metric),
                           stress_energy, ricci, ricci_scalar,
                           inverse_spacetime_metric, induced_spatial_metric);
  const Approx custom_approx = Approx::custom().epsilon(2.e-13).scale(1.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      CHECK_ITERABLE_CUSTOM_APPROX(electric_weyl.get(i, j),
                                   (DataVector{num_points, 0.0}),
                                   custom_approx);
    }
  }
  TestHelpers::db::test_compute_tag<
      hydro::Tags::WeylElectricCompute<DataVector>>("WeylElectric");
  TestHelpers::db::test_compute_tag<
      hydro::Tags::WeylElectricScalarCompute<DataVector>>("WeylElectricScalar");
}

template <typename DataType>
void test_curvature_quantities_random_values(const DataType& used_for_size) {
  tnsr::aa<DataType, 3> (*f_ricci_tensor)(const tnsr::AA<DataType, 3>&,
                                          const tnsr::aa<DataType, 3>&) =
      &hydro::ricci_in_gr<DataType>;
  pypp::check_with_random_values<1>(f_ricci_tensor, "CurvatureQuantities",
                                    "ricci_in_gr", {{{-1., 1.}}},
                                    used_for_size);

  Scalar<DataType> (*f_ricci_scalar)(const tnsr::aa<DataType, 3>&,
                                     const tnsr::AA<DataType, 3>&) =
      &gr::ricci_scalar<3, Frame::Inertial, IndexType::Spacetime, DataType>;
  pypp::check_with_random_values<1>(f_ricci_scalar, "CurvatureQuantities",
                                    "ricci_scalar", {{{-1., 1.}}},
                                    used_for_size);

  tnsr::ii<DataType, 3> (*f_weyl_electric)(
      const tnsr::ii<DataType, 3>&, const tnsr::AA<DataType, 3>&,
      const tnsr::aa<DataType, 3>&, const Scalar<DataType>&,
      const tnsr::AA<DataType, 3>&, const tnsr::aa<DataType, 3>&) =
      &hydro::weyl_electric<DataType>;
  pypp::check_with_random_values<1>(f_weyl_electric, "CurvatureQuantities",
                                    "weyl_electric", {{{-1., 1.}}},
                                    used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.CurvatureQuantities",
                  "[PointwiseFunctions][Unit]") {
  const pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");
  test_curvature_quantities_flrw();
  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_curvature_quantities_random_values,
                                    ());
}
