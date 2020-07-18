// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"

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
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace InitializeJ {
namespace detail {
// perform an iterative solve for the set of angular coordinates necessary to
// set the gauge transformed version of `surface_j` to zero. This reliably
// converges eventually provided `surface_j` is initially reasonably small. As a
// comparatively primitive method, the convergence often takes several
// iterations (10-100) to reach roundoff; However, the iterations are fast, and
// the computation is for initial data that needs to be computed only once
// during a simulation, so it is not currently an optimization priority. If this
// function becomes a bottleneck, the numerical procedure of the iterative
// method should be revisited.
double adjust_angular_coordinates_for_j(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const SpinWeighted<ComplexDataVector, 2>& surface_j, const size_t l_max,
    const double tolerance, const size_t max_steps,
    const bool adjust_volume_gauge) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get<0>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.theta;
    get<1>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.phi;
  }
  get<0>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      cos(get<1>(*angular_cauchy_coordinates));
  get<1>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      sin(get<1>(*angular_cauchy_coordinates));
  get<2>(*cartesian_cauchy_coordinates) =
      cos(get<0>(*angular_cauchy_coordinates));

  Variables<tmpl::list<
      // cartesian coordinates
      ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // eth of cartesian coordinates
      ::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      // eth of gauge-transformed cartesian coordinates
      ::Tags::SpinWeighted<::Tags::TempScalar<6, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<7, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<8, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      // iterated J
      ::Tags::SpinWeighted<::Tags::TempScalar<9, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      // intermediate J buffer
      ::Tags::SpinWeighted<::Tags::TempScalar<10, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      // K buffer
      ::Tags::SpinWeighted<::Tags::TempScalar<11, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // gauge Jacobians
      ::Tags::SpinWeighted<::Tags::TempScalar<12, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<13, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // gauge Jacobians on next iteration
      ::Tags::SpinWeighted<::Tags::TempScalar<14, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<15, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // gauge conformal factor
      ::Tags::SpinWeighted<::Tags::TempScalar<16, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // cartesian coordinates steps
      ::Tags::SpinWeighted<::Tags::TempScalar<17, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<18, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<19, ComplexDataVector>,
                           std::integral_constant<int, 0>>>>
      computation_buffers{number_of_angular_points};

  auto& x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));

  x.data() =
      std::complex<double>(1.0, 0.0) * get<0>(*cartesian_cauchy_coordinates);
  y.data() =
      std::complex<double>(1.0, 0.0) * get<1>(*cartesian_cauchy_coordinates);
  z.data() =
      std::complex<double>(1.0, 0.0) * get<2>(*cartesian_cauchy_coordinates);

  auto& eth_x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));

  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Eth,
                 Spectral::Swsh::Tags::Eth>>(l_max, 1, make_not_null(&eth_x),
                                             make_not_null(&eth_y),
                                             make_not_null(&eth_z), x, y, z);

  auto& evolution_gauge_eth_x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<6, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& evolution_gauge_eth_y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<7, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& evolution_gauge_eth_z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<8, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));

  auto& evolution_gauge_surface_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<9, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          computation_buffers));

  auto& interpolated_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<10, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          computation_buffers));
  auto& interpolated_k =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<11, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));

  auto& gauge_c =
      get<::Tags::SpinWeighted<::Tags::TempScalar<12, ComplexDataVector>,
                               std::integral_constant<int, 2>>>(
          computation_buffers);
  auto& gauge_d =
      get<::Tags::SpinWeighted<::Tags::TempScalar<13, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& next_gauge_c =
      get<::Tags::SpinWeighted<::Tags::TempScalar<14, ComplexDataVector>,
                               std::integral_constant<int, 2>>>(
          computation_buffers);
  auto& next_gauge_d =
      get<::Tags::SpinWeighted<::Tags::TempScalar<15, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& gauge_omega =
      get<::Tags::SpinWeighted<::Tags::TempScalar<16, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& x_step =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<17, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& y_step =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<18, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& z_step =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<19, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));

  double max_error = 1.0;
  size_t number_of_steps = 0;
  while (true) {
    GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                            cartesian_cauchy_coordinates);

    Spectral::Swsh::SwshInterpolator iteration_interpolator{
      get<0>(*angular_cauchy_coordinates),
      get<1>(*angular_cauchy_coordinates), l_max};

    GaugeUpdateJacobianFromCoordinates<
        Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                            make_not_null(&gauge_d),
                                            angular_cauchy_coordinates,
                                            *cartesian_cauchy_coordinates,
                                            l_max);

    iteration_interpolator.interpolate(make_not_null(&interpolated_j),
                                       surface_j);
    interpolated_k.data() =
        sqrt(1.0 + interpolated_j.data() * conj(interpolated_j.data()));

    get(gauge_omega).data() =
        0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                   get(gauge_c).data() * conj(get(gauge_c).data()));
    // the fully computed j in the coordinate system determined so far
    // (`previous_angular_cauchy_coordinates`)
    evolution_gauge_surface_j.data() =
        0.25 *
        (square(conj(get(gauge_d).data())) * interpolated_j.data() +
         square(get(gauge_c).data()) * conj(interpolated_j.data()) +
         2.0 * get(gauge_c).data() * conj(get(gauge_d).data()) *
             interpolated_k.data()) /
        square(get(gauge_omega).data());

    // check completion conditions
    max_error = max(abs(evolution_gauge_surface_j.data()));
    ++number_of_steps;
    if (max_error > 5.0e-3) {
      ERROR(
          "Iterative solve for surface coordinates of initial data failed. The "
          "strain is too large to be fully eliminated by a well-behaved "
          "alteration of the spherical mesh. For this data, please use an "
          "alternative initial data generator such as "
          "`InitializeJInverseCubic`.");
    }
    if (max_error < tolerance or number_of_steps > max_steps) {
      break;
    }
    // The alteration in each of the spin-weighted Jacobian factors determined
    // by linearizing the system in small J
    get(next_gauge_c).data() = -0.5 * evolution_gauge_surface_j.data() *
                               square(get(gauge_omega).data()) /
                               (get(gauge_d).data() * interpolated_k.data());
    get(next_gauge_d).data() = get(next_gauge_c).data() *
                               conj(get(gauge_c).data()) /
                               conj(get(gauge_d).data());

    iteration_interpolator.interpolate(make_not_null(&evolution_gauge_eth_x),
                                       eth_x);
    iteration_interpolator.interpolate(make_not_null(&evolution_gauge_eth_y),
                                       eth_y);
    iteration_interpolator.interpolate(make_not_null(&evolution_gauge_eth_z),
                                       eth_z);

    evolution_gauge_eth_x =
        0.5 * ((get(next_gauge_c)) * conj(evolution_gauge_eth_x) +
               conj((get(next_gauge_d))) * evolution_gauge_eth_x);
    evolution_gauge_eth_y =
        0.5 * ((get(next_gauge_c)) * conj(evolution_gauge_eth_y) +
               conj((get(next_gauge_d))) * evolution_gauge_eth_y);
    evolution_gauge_eth_z =
        0.5 * ((get(next_gauge_c)) * conj(evolution_gauge_eth_z) +
               conj((get(next_gauge_d))) * evolution_gauge_eth_z);

    // here we attempt to just update the current value according to the
    // alteration suggested. Ideally that's the `dominant` part of the needed
    // alteration
    Spectral::Swsh::angular_derivatives<tmpl::list<
        Spectral::Swsh::Tags::InverseEth, Spectral::Swsh::Tags::InverseEth,
        Spectral::Swsh::Tags::InverseEth>>(
        l_max, 1, make_not_null(&x_step), make_not_null(&y_step),
        make_not_null(&z_step), evolution_gauge_eth_x, evolution_gauge_eth_y,
        evolution_gauge_eth_z);

    get<0>(*cartesian_cauchy_coordinates) += real(x_step.data());
    get<1>(*cartesian_cauchy_coordinates) += real(y_step.data());
    get<2>(*cartesian_cauchy_coordinates) += real(z_step.data());
  }

  GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords,
      Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                          cartesian_cauchy_coordinates);

  if (adjust_volume_gauge) {
    GaugeUpdateJacobianFromCoordinates<
        Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                            make_not_null(&gauge_d),
                                            angular_cauchy_coordinates,
                                            *cartesian_cauchy_coordinates,
                                            l_max);

    get(gauge_omega).data() =
        0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                   get(gauge_c).data() * conj(get(gauge_c).data()));

    GaugeAdjustInitialJ::apply(volume_j, gauge_c, gauge_d, gauge_omega,
                               *angular_cauchy_coordinates, l_max);
  }
  return max_error;
}
}  // namespace detail

void GaugeAdjustInitialJ::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_omega,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
        cauchy_angular_coordinates,
    const size_t l_max) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(*volume_j).size() /
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Scalar<SpinWeighted<ComplexDataVector, 2>> evolution_coords_j_buffer{
      number_of_angular_points};
  Spectral::Swsh::SwshInterpolator interpolator{
      get<0>(cauchy_angular_coordinates), get<1>(cauchy_angular_coordinates),
      l_max};
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
}  // namespace InitializeJ
}  // namespace Cce
