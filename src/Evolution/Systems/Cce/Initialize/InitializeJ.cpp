// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ {
namespace detail {
void unit_sphere_eth_cartesian_coordinates(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> eth_x,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> eth_y,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> eth_z,
    const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Variables<
      tmpl::list<::Tags::Tempi<0, 3>,
                 ::Tags::Tempi<1, 2, ::Frame::Spherical<::Frame::Inertial>>>>
      coordinate_buffers{number_of_angular_points};
  auto& [cartesian_coordinates, angular_coordinates] = coordinate_buffers;
  Spectral::Swsh::create_angular_and_cartesian_coordinates(
      make_not_null(&cartesian_coordinates),
      make_not_null(&angular_coordinates), l_max);

  // the Cartesian coordinates as spin-weight-0 functions, to take their eth
  Variables<tmpl::list<::Tags::TempSpinWeightedScalar<0, 0>,
                       ::Tags::TempSpinWeightedScalar<1, 0>,
                       ::Tags::TempSpinWeightedScalar<2, 0>>>
      swsh_buffers{number_of_angular_points};
  auto& [x, y, z] = swsh_buffers;
  get(x).data() =
      std::complex<double>(1.0, 0.0) * get<0>(cartesian_coordinates);
  get(y).data() =
      std::complex<double>(1.0, 0.0) * get<1>(cartesian_coordinates);
  get(z).data() =
      std::complex<double>(1.0, 0.0) * get<2>(cartesian_coordinates);
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Eth,
                 Spectral::Swsh::Tags::Eth>>(l_max, 1, eth_x, eth_y, eth_z,
                                             get(x), get(y), get(z));
}

void update_angular_coordinates_and_jacobians(
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const gsl::not_null<Spectral::Swsh::SwshInterpolator*> interpolator,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> gauge_c,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> gauge_d,
    const size_t l_max) {
  GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords,
      Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                          cartesian_cauchy_coordinates);
  *interpolator = Spectral::Swsh::SwshInterpolator{
      get<0>(*angular_cauchy_coordinates), get<1>(*angular_cauchy_coordinates),
      l_max};
  GaugeUpdateJacobianFromCoordinates<
      Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Tags::CauchyAngularCoords,
      Tags::CauchyCartesianCoords>::apply(gauge_c, gauge_d,
                                          *angular_cauchy_coordinates,
                                          *cartesian_cauchy_coordinates, l_max);
}

ResidualPlateau::ResidualPlateau(const size_t window) : window_{window} {}

void ResidualPlateau::record(const double residual) {
  smallest_residual_.push_back(
      smallest_residual_.empty()
          ? residual
          : std::min(residual, smallest_residual_.back()));
}

bool ResidualPlateau::stopped_improving(const double tolerance) const {
  const size_t number_of_passes = smallest_residual_.size();
  if (number_of_passes <= window_) {
    return false;
  }
  const double earlier = smallest_residual_[number_of_passes - 1 - window_];
  const double improvement =
      earlier > 0.0 ? 1.0 - smallest_residual_.back() / earlier : 0.0;
  return improvement < tolerance;
}

void check_angular_solve_error_threshold(const double max_error,
                                         const double error_threshold) {
  if (std::isnan(max_error) or max_error > error_threshold) {
    ERROR(
        "Iterative solve for surface coordinates of initial data failed. The "
        "strain is too large to be fully eliminated by a well-behaved "
        "alteration of the spherical mesh. This could be an indication that "
        "there is an issue with the worldtube data. If you are confident "
        "the worldtube data is correct, then please use an alternative "
        "initial data generator such as `InverseCubic`. If that fails, "
        "please double check that your spherical harmonic modes are decaying "
        "correctly with increasing (l,m).\nError: "
        << max_error << "\nError threshold: " << error_threshold);
  }
}

void report_unconverged_angular_solve(
    const UnconvergedAngularSolve if_unconverged, const double tolerance,
    const size_t number_of_iterations, const double max_error) {
  switch (if_unconverged) {
    case UnconvergedAngularSolve::Error:
      ERROR(
          "Initial data iterative angular solve did not reach "
          "target tolerance "
          << tolerance << ".\n"
          << "Exited after " << number_of_iterations
          << " iterations, achieving final\n"
             "maximum over collocation points deviation of J from target of "
          << max_error);
    case UnconvergedAngularSolve::Warn:
      Parallel::printf(
          "Warning: iterative angular solve did not reach "
          "target tolerance %e.\n"
          "Exited after %zu iterations, achieving final maximum over "
          "collocation points for deviation from target of %e\n"
          "Proceeding with evolution using the partial result from partial "
          "angular solve.\n",
          tolerance, number_of_iterations, max_error);
      return;
    case UnconvergedAngularSolve::Silent:
      return;
    case UnconvergedAngularSolve::Uninitialized:
      ERROR("UnconvergedAngularSolve was not set; pass Error, Warn or Silent.");
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("An unknown value of UnconvergedAngularSolve was passed: "
            << static_cast<int>(if_unconverged));
      // LCOV_EXCL_STOP
  }
}
}  // namespace detail

void GaugeAdjustInitialJ::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_omega,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
    /*cauchy_angular_coordinates*/,
    const Spectral::Swsh::SwshInterpolator& interpolator, const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(*volume_j).size() /
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Scalar<SpinWeighted<ComplexDataVector, 2>> evolution_coords_j_buffer{
      number_of_angular_points};
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    Scalar<SpinWeighted<ComplexDataVector, 2>> angular_view_j;
    get(angular_view_j)
        .set_data_ref(
            get(*volume_j).data().data() + i * number_of_angular_points,
            number_of_angular_points);
    get(evolution_coords_j_buffer) = get(angular_view_j);
    GaugeAdjustedBoundaryValue<Tags::BondiJ>::apply(
        make_not_null(&angular_view_j), evolution_coords_j_buffer, gauge_c,
        gauge_d, gauge_omega, interpolator);
  }
}
}  // namespace Cce::InitializeJ
