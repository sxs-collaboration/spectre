// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/ApparentHorizons/StrahlkorperGrTestHelpers.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataVector.hpp"                  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace TestHelpers {
namespace Schwarzschild {

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> spatial_ricci(
    const tnsr::I<DataType, SpatialDim, Frame>& x,
    const double& mass) noexcept {
  auto ricci = make_with_value<tnsr::ii<DataType, SpatialDim, Frame>>(x, 0.);

  constexpr auto dimensionality = index_dim<0>(ricci);

  const DataType r = get(magnitude(x));

  for (size_t i = 0; i < dimensionality; ++i) {
    for (size_t j = i; j < dimensionality; ++j) {
      ricci.get(i, j) -= (8.0 * mass + 3.0 * r) * x.get(i) * x.get(j);
      if (i == j) {
        ricci.get(i, j) += square(r) * (4.0 * mass + r);
      }
      ricci.get(i, j) *= mass;
      ricci.get(i, j) /= pow<4>(r) * square(2.0 * mass + r);
    }
  }

  return ricci;
}
}  // namespace Schwarzschild

namespace Minkowski {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature_sphere(
    const tnsr::I<DataType, SpatialDim, Frame>& x) noexcept {
  auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataType, SpatialDim, Frame>>(x, 0.);

  constexpr auto dimensionality = index_dim<0>(extrinsic_curvature);

  const DataType one_over_r = 1.0 / get(magnitude(x));

  for (size_t i = 0; i < dimensionality; ++i) {
    extrinsic_curvature.get(i, i) += 1.0;
    for (size_t j = i; j < dimensionality; ++j) {
      extrinsic_curvature.get(i, j) -= x.get(i) * x.get(j) * square(one_over_r);
      extrinsic_curvature.get(i, j) *= one_over_r;
    }
  }

  return extrinsic_curvature;
}
}  // namespace Minkowski

namespace Kerr {
template <typename DataType>
Scalar<DataType> horizon_ricci_scalar(
    const Scalar<DataType>& horizon_radius, const double& mass,
    const double& dimensionless_spin_z) noexcept {
  // Compute Kerr spin parameter a
  // This is the magnitude of the dimensionless spin times the mass
  double kerr_spin_a = mass * dimensionless_spin_z;

  // Compute the Boyer-Lindquist horizon radius, r+
  const double kerr_r_plus = mass + sqrt(square(mass) - square(kerr_spin_a));

  // Get the Ricci scalar of the horizon, e.g. Eq. (119) of
  // https://arxiv.org/abs/0706.0622
  // The precise relation used here is derived in
  // https://v2.overleaf.com/read/twdtxchyrtyv
  Scalar<DataVector> ricci_scalar(
      2.0 * (square(kerr_r_plus) + square(kerr_spin_a)) *
      (3.0 * square(get(horizon_radius)) - 2.0 * square(kerr_r_plus) -
       3.0 * square(kerr_spin_a)));
  get(ricci_scalar) /= cube(-1.0 * square(get(horizon_radius)) +
                               square(kerr_spin_a) + 2.0 * square(kerr_r_plus));
  return ricci_scalar;
}

template <typename DataType>
Scalar<DataType> horizon_ricci_scalar(
    const Scalar<DataType>& horizon_radius_with_spin_on_z_axis,
    const YlmSpherepack& ylm_with_spin_on_z_axis, const YlmSpherepack& ylm,
    const double& mass,
    const std::array<double, 3>& dimensionless_spin) noexcept {
  // get the dimensionless spin magnitude and direction
  const double spin_magnitude = magnitude(dimensionless_spin);
  const double spin_theta =
      atan2(sqrt(square(dimensionless_spin[0]) + square(dimensionless_spin[1])),
            dimensionless_spin[2]);

  // Return the aligned-spin result if spin is close enough to the z axis,
  // to avoid a floating-point exception. The choice of eps here is arbitrary.
  const double eps = 1.e-10;

  // There are 2 YlmSpherepacks: i) ylm, for the actual black hole, with spin
  // in a generic direction, and ii) ylm_with_spin_on_z_axis, for a black hole
  // with the same spin magnitude but with the spin in the +z direction.
  // To get the horizon Ricci scalar for the actual black hole, do this:
  //    1. Find the horizon Ricci scalar for the aligned spin
  //    2. Let the generic spin point in direction (spin_theta, spin_phi).
  //       Rotate the ylm.theta_phi_points by -spin_phi about the z axis and
  //       then by -spin_theta about the y axis, so the point
  //       (spin_theta, spin_phi) is mapped to (0, 0), the +z axis.
  //    3. Interpolate the aligned-spin Ricci scalar from step 1 at each
  //       rotated point from step 2 to get the horizon Ricci scalar
  //       for the corresponding (unrotated) ylm_theta_phi_points.

  // Get the ricci scalar for a Kerr black hole with spin in the +z direction
  // but same mass and spin magnitude
  const auto ricci_scalar_with_spin_on_z_axis = horizon_ricci_scalar(
      horizon_radius_with_spin_on_z_axis, mass, spin_magnitude);

  // Is the spin aligned? If so, just return the aligned-spin scalar curvature
  if (abs(spin_theta) < eps or abs(spin_theta - M_PI) < eps) {
    return ricci_scalar_with_spin_on_z_axis;
  }

  const double spin_phi = atan2(dimensionless_spin[1], dimensionless_spin[0]);

  // Get the theta and phi points on the original Strahlkorper, where the
  // spin is not on the z axis
  const auto& theta_phi_points = ylm.theta_phi_points();

  // Loop over collocation points, rotating each point
  std::vector<std::array<double, 2>> points;
  for (size_t i = 0; i < get(ricci_scalar_with_spin_on_z_axis).size(); ++i) {
    // Get (theta,phi) of the collocation point on the actual horizon
    const double theta = theta_phi_points[0][i];
    const double phi = theta_phi_points[1][i];

    // Rotate the coordinates on the original Strahlkorper so that a point
    // on the spin axis gets mapped to a point on the +z axis.
    // This means the new coordinates are rotated from the old ones by
    // -spin_phi about the z axis and then by -spin_theta about the y axis.
    // The unrotated x,y,z coordinates are defined on the unit sphere:
    // x = sin(theta)*cos(phi), y = sin(theta) * sin(phi), z = cos(theta)

    const double x_new = cos(spin_theta) * cos(phi - spin_phi) * sin(theta) -
                         cos(theta) * sin(spin_theta);
    const double y_new = sin(theta) * sin(phi - spin_phi);
    const double z_new = cos(theta) * cos(spin_theta) +
                         cos(phi - spin_phi) * sin(theta) * sin(spin_theta);

    // Since I'm rotating on the unit sphere, the radius of the unrotated and
    // new points is unity.
    const double theta_new = atan2(sqrt(square(x_new) + square(y_new)), z_new);
    double phi_new = (abs(theta_new) > eps and abs(theta_new - M_PI) > eps)
                         ? atan2(y_new, x_new)
                         : 0.0;
    // Ensure phi_new is between 0 and 2 pi.
    if (phi_new < 0.0) {
      phi_new += 2.0 * M_PI;
    }

    // Add the point to the list of points to interpolate to
    points.emplace_back(std::array<double, 2>{{theta_new, phi_new}});
  }

  // Interpolate ricci_scalar_with_spin_on_z_axis onto the new points
  auto interpolation_info =
      ylm_with_spin_on_z_axis.set_up_interpolation_info(points);
  std::vector<double> ricci_scalar_interpolated(interpolation_info.size());
  ylm_with_spin_on_z_axis.interpolate(
      &ricci_scalar_interpolated, get(ricci_scalar_with_spin_on_z_axis).data(),
      interpolation_info);

  // Load the interpolated values into the DataVector ricci_scalar
  Scalar<DataVector> ricci_scalar =
      make_with_value<Scalar<DataVector>>(theta_phi_points[0], 0.0);
  for (size_t i = 0; i < get(ricci_scalar_with_spin_on_z_axis).size(); ++i) {
    get(ricci_scalar)[i] = ricci_scalar_interpolated[i];
  }

  return ricci_scalar;
}

}  // namespace Kerr
}  // namespace TestHelpers

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE(_, data)                                 \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>     \
  TestHelpers::Schwarzschild::spatial_ricci(                 \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x, \
      const double& mass) noexcept;                          \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>     \
  TestHelpers::Minkowski::extrinsic_curvature_sphere(        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INDEXTYPE
#undef INSTANTIATE

template Scalar<DataVector> TestHelpers::Kerr::horizon_ricci_scalar(
    const Scalar<DataVector>& horizon_radius, const double& mass,
    const double& dimensionless_spin_z) noexcept;
template Scalar<DataVector> TestHelpers::Kerr::horizon_ricci_scalar(
    const Scalar<DataVector>& horizon_radius_with_spin_on_z_axis,
    const YlmSpherepack& ylm_with_spin_on_z_axis, const YlmSpherepack& ylm,
    const double& mass, const std::array<double, 3>& spin) noexcept;
