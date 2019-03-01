// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Error.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrSchildCoords.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/Pypp.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <typename DataType>
void test_kerr_schild_coords(const DataType& used_for_size) noexcept {
  const double bh_mass = 2.54;
  const double bh_dimless_spin = 0.9375;
  const auto member_variables = std::make_tuple(bh_mass, bh_dimless_spin);
  gr::KerrSchildCoords kerr_schild_coords(bh_mass, bh_dimless_spin);

  pypp::check_with_random_values<1>(
      &gr::KerrSchildCoords::r_coord_squared<DataType>, kerr_schild_coords,
      "KerrSchildCoords", "r_coord_squared", {{{-10.0, 10.0}}},
      member_variables, used_for_size);

  pypp::check_with_random_values<1>(
      &gr::KerrSchildCoords::cartesian_from_spherical_ks<DataType>,
      kerr_schild_coords, "KerrSchildCoords", "cartesian_from_spherical_ks",
      {{{-10.0, 10.0}}}, member_variables, used_for_size);
}

template <typename DataType>
void test_coord_transformation_on_xy_plane(
    const DataType& used_for_size) noexcept {
  const double bh_mass = 0.34;
  const double bh_dimless_spin = 0.6332;
  const double spin_a = bh_mass * bh_dimless_spin;

  MAKE_GENERATOR(generator);

  // get random point outside of the ring singularity x^2 + y^2 = a^2
  // to avoid possible FPE when computing the Jacobian.
  std::uniform_real_distribution<> distribution_mod(spin_a, 10);
  const double random_mod = distribution_mod(generator);
  std::uniform_real_distribution<> distribution_angle(0, 2.0 * M_PI);
  const double random_angle = distribution_angle(generator);
  auto random_point_on_xy_plane =
      make_with_value<tnsr::I<DataType, 3>>(used_for_size, 0.0);
  get<0>(random_point_on_xy_plane) = random_mod * cos(random_angle);
  get<1>(random_point_on_xy_plane) = random_mod * sin(random_angle);

  std::uniform_real_distribution<> distribution_vector(-3.0, 3.0);
  auto random_vector_to_transform =
      make_with_value<tnsr::I<DataType, 3, Frame::NoFrame>>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    random_vector_to_transform.get(i) = distribution_vector(generator);
  }

  CHECK_ITERABLE_APPROX(
      (gr::KerrSchildCoords{bh_mass, bh_dimless_spin}
           .cartesian_from_spherical_ks(random_vector_to_transform,
                                        random_point_on_xy_plane)),
      (pypp::call<tnsr::I<DataType, 3>>(
          "KerrSchildCoords", "cartesian_from_spherical_ks",
          random_vector_to_transform, random_point_on_xy_plane, bh_mass,
          bh_dimless_spin)));
}

template <typename DataType>
void test_coord_transformation_along_z_axis(
    const DataType& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-13.0, 13.0);

  auto random_radial_vector_to_transform =
      make_with_value<tnsr::I<DataType, 3, Frame::NoFrame>>(used_for_size, 0.0);
  get<0>(random_radial_vector_to_transform) = distribution(generator);

  const double z_coordinate = distribution(generator);
  auto random_point_along_z_axis =
      make_with_value<tnsr::I<DataType, 3>>(used_for_size, 0.0);
  get<2>(random_point_along_z_axis) = z_coordinate;

  gr::KerrSchildCoords ks_coords{1.532, 0.9375};

  // New vector should point along z-axis, with v^z = sign(z) v^r
  const auto transformed_vector = ks_coords.cartesian_from_spherical_ks(
      random_radial_vector_to_transform, random_point_along_z_axis);
  CHECK(get<0>(transformed_vector) == 0.0);
  CHECK(get<1>(transformed_vector) == 0.0);
  CHECK(get<2>(transformed_vector) ==
        (z_coordinate > 0.0 ? 1.0 : -1.0) *
            get<0>(random_radial_vector_to_transform));

  // Since we transform a uniform vector, the transformation should not modify
  // the new vector upon flipping the sign of the z coordinate
  get<2>(random_point_along_z_axis) *= -1.0;
  // v^r flips sign in order to represent the same uniform vector at z < 0
  get<0>(random_radial_vector_to_transform) *= -1.0;
  CHECK_ITERABLE_APPROX(
      transformed_vector,
      ks_coords.cartesian_from_spherical_ks(random_radial_vector_to_transform,
                                            random_point_along_z_axis));
}

template <typename DataType>
void test_theta_component(const DataType& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-13.0, 13.0);

  auto random_point_along_z_axis =
      make_with_value<tnsr::I<DataType, 3>>(used_for_size, 0.0);
  get<2>(random_point_along_z_axis) = distribution(generator);

  // input vector has a nonvanishing theta component along the z-axis.
  gr::KerrSchildCoords{3.123, 0.854}.cartesian_from_spherical_ks(
      make_with_value<tnsr::I<DataType, 3, Frame::NoFrame>>(
          used_for_size, distribution(generator)),
      random_point_along_z_axis);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.KerrSchildCoords",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  const double d(std::numeric_limits<double>::signaling_NaN());
  const DataVector dv(5);
  test_kerr_schild_coords(d);
  test_kerr_schild_coords(dv);
  test_coord_transformation_along_z_axis(d);
  test_coord_transformation_along_z_axis(dv);
  test_coord_transformation_on_xy_plane(d);
  test_coord_transformation_on_xy_plane(dv);

  gr::KerrSchildCoords kerr_schild_coords(0.8624, 0.151);
  test_serialization(kerr_schild_coords);

  gr::KerrSchildCoords kerr_schild_coords_copy(0.8624, 0.151);
  test_move_semantics(std::move(kerr_schild_coords),
                      kerr_schild_coords_copy);  // NOLINT
}

// [[OutputRegex, The input vector must have a vanishing theta component]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.KerrSchildCoords.AlongZDouble",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  test_theta_component(std::numeric_limits<double>::signaling_NaN());
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The input vector must have a vanishing theta component]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.KerrSchildCoords.AlongZDv",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  test_theta_component(DataVector(5));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The mass must be positive]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.KerrSchildCoords.Mass",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  gr::KerrSchildCoords ks_coords(-4.21, 0.999);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The dimensionless spin must be in the range \(-1, 1\)]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.KerrSchildCoords.SpinLower",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  gr::KerrSchildCoords ks_coords(0.15, -1.3);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The dimensionless spin must be in the range \(-1, 1\)]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.KerrSchildCoords.SpinUpper",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  gr::KerrSchildCoords ks_coords(1.532, 4.2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
