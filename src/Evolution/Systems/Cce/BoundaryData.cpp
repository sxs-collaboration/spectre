// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/BoundaryData.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Math.hpp"

namespace Cce {

void trigonometric_functions_on_swsh_collocation(
    const gsl::not_null<Scalar<DataVector>*> cos_phi,
    const gsl::not_null<Scalar<DataVector>*> cos_theta,
    const gsl::not_null<Scalar<DataVector>*> sin_phi,
    const gsl::not_null<Scalar<DataVector>*> sin_theta,
    const size_t l_max) noexcept {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  destructive_resize_components(cos_phi, size);
  destructive_resize_components(cos_theta, size);
  destructive_resize_components(sin_phi, size);
  destructive_resize_components(sin_theta, size);

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get(*sin_theta)[collocation_point.offset] = sin(collocation_point.theta);
    get(*cos_theta)[collocation_point.offset] = cos(collocation_point.theta);
    get(*sin_phi)[collocation_point.offset] = sin(collocation_point.phi);
    get(*cos_phi)[collocation_point.offset] = cos(collocation_point.phi);
  }
}

void cartesian_to_spherical_coordinates_and_jacobians(
    const gsl::not_null<tnsr::I<DataVector, 3>*> unit_cartesian_coords,
    const gsl::not_null<SphericaliCartesianJ*> cartesian_to_spherical_jacobian,
    const gsl::not_null<CartesianiSphericalJ*>
        inverse_cartesian_to_spherical_jacobian,
    const Scalar<DataVector>& cos_phi, const Scalar<DataVector>& cos_theta,
    const Scalar<DataVector>& sin_phi, const Scalar<DataVector>& sin_theta,
    const double extraction_radius) noexcept {
  const size_t size = get(cos_phi).size();
  destructive_resize_components(unit_cartesian_coords, size);
  destructive_resize_components(cartesian_to_spherical_jacobian, size);
  destructive_resize_components(inverse_cartesian_to_spherical_jacobian, size);

  // note: factor of r scaled out
  get<0>(*unit_cartesian_coords) = get(sin_theta) * get(cos_phi);
  get<1>(*unit_cartesian_coords) = get(sin_theta) * get(sin_phi);
  get<2>(*unit_cartesian_coords) = get(cos_theta);

  // dx/dr   dy/dr  dz/dr
  get<0, 0>(*cartesian_to_spherical_jacobian) = get(sin_theta) * get(cos_phi);
  get<0, 1>(*cartesian_to_spherical_jacobian) = get(sin_theta) * get(sin_phi);
  get<0, 2>(*cartesian_to_spherical_jacobian) = get(cos_theta);
  // dx/dtheta   dy/dtheta  dz/dtheta
  get<1, 0>(*cartesian_to_spherical_jacobian) =
      extraction_radius * get(cos_theta) * get(cos_phi);
  get<1, 1>(*cartesian_to_spherical_jacobian) =
      extraction_radius * get(cos_theta) * get(sin_phi);
  get<1, 2>(*cartesian_to_spherical_jacobian) =
      -extraction_radius * get(sin_theta);
  // (1/sin(theta)) { dx/dphi,   dy/dphi,  dz/dphi }
  get<2, 0>(*cartesian_to_spherical_jacobian) =
      -extraction_radius * get(sin_phi);
  get<2, 1>(*cartesian_to_spherical_jacobian) =
      extraction_radius * get(cos_phi);
  get<2, 2>(*cartesian_to_spherical_jacobian) = 0.0;

  // dr/dx   dtheta/dx   dphi/dx * sin(theta)
  get<0, 0>(*inverse_cartesian_to_spherical_jacobian) =
      get(cos_phi) * get(sin_theta);
  get<0, 1>(*inverse_cartesian_to_spherical_jacobian) =
      get(cos_phi) * get(cos_theta) / extraction_radius;
  get<0, 2>(*inverse_cartesian_to_spherical_jacobian) =
      -get(sin_phi) / (extraction_radius);
  // dr/dy   dtheta/dy   dphi/dy * sin(theta)
  get<1, 0>(*inverse_cartesian_to_spherical_jacobian) =
      get(sin_phi) * get(sin_theta);
  get<1, 1>(*inverse_cartesian_to_spherical_jacobian) =
      get(cos_theta) * get(sin_phi) / extraction_radius;
  get<1, 2>(*inverse_cartesian_to_spherical_jacobian) =
      get(cos_phi) / (extraction_radius);
  // dr/dz   dtheta/dz   dphi/dz * sin(theta)
  get<2, 0>(*inverse_cartesian_to_spherical_jacobian) = get(cos_theta);
  get<2, 1>(*inverse_cartesian_to_spherical_jacobian) =
      -get(sin_theta) / extraction_radius;
  get<2, 2>(*inverse_cartesian_to_spherical_jacobian) = 0.0;
}

void cartesian_spatial_metric_and_derivatives_from_modes(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> cartesian_spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, 3>*>
        inverse_cartesian_spatial_metric,
    const gsl::not_null<tnsr::ijj<DataVector, 3>*> d_cartesian_spatial_metric,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> dt_cartesian_spatial_metric,
    const gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const tnsr::ii<ComplexModalVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dt_spatial_metric_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const size_t l_max) noexcept {
  const size_t size = get<0, 0>(inverse_cartesian_to_spherical_jacobian).size();
  destructive_resize_components(cartesian_spatial_metric, size);
  destructive_resize_components(d_cartesian_spatial_metric, size);
  destructive_resize_components(dt_cartesian_spatial_metric, size);

  destructive_resize_components(interpolation_buffer, size);
  destructive_resize_components(interpolation_modal_buffer, size);
  destructive_resize_components(eth_buffer, size);

  // Allocation
  SphericaliCartesianjj spherical_d_cartesian_spatial_metric{size};

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      // copy the modes to a spin-weighted type for interpolation
      get(*interpolation_modal_buffer).data() =
          spatial_metric_coefficients.get(i, j);
      Spectral::Swsh::inverse_swsh_transform(
          l_max, 1, make_not_null(&get(*interpolation_buffer)),
          get(*interpolation_modal_buffer));
      cartesian_spatial_metric->get(i, j) =
          real(get(*interpolation_buffer).data());

      get(*interpolation_modal_buffer).data() =
          dt_spatial_metric_coefficients.get(i, j);
      Spectral::Swsh::inverse_swsh_transform(
          l_max, 1, make_not_null(&get(*interpolation_buffer)),
          get(*interpolation_modal_buffer));
      dt_cartesian_spatial_metric->get(i, j) =
          real(get(*interpolation_buffer).data());

      get(*interpolation_modal_buffer).data() =
          dr_spatial_metric_coefficients.get(i, j);
      Spectral::Swsh::inverse_swsh_transform(
          l_max, 1, make_not_null(&get(*interpolation_buffer)),
          get(*interpolation_modal_buffer));
      spherical_d_cartesian_spatial_metric.get(0, i, j) =
          real(get(*interpolation_buffer).data());
    }
  }

  *inverse_cartesian_spatial_metric =
      determinant_and_inverse(*cartesian_spatial_metric).second;

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      // reusing the interpolation buffer for taking the angular derivatives
      get(*interpolation_buffer) =
          std::complex<double>(1.0, 0.0) * cartesian_spatial_metric->get(i, j);
      Spectral::Swsh::angular_derivatives<
          tmpl::list<Spectral::Swsh::Tags::Eth>>(
          l_max, 1, make_not_null(&get(*eth_buffer)),
          get(*interpolation_buffer));
      spherical_d_cartesian_spatial_metric.get(1, i, j) =
          -real(get(*eth_buffer).data());
      spherical_d_cartesian_spatial_metric.get(2, i, j) =
          -imag(get(*eth_buffer).data());
    }
  }

  // convert derivatives to cartesian form
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        d_cartesian_spatial_metric->get(k, i, j) =
            inverse_cartesian_to_spherical_jacobian.get(k, 0) *
            spherical_d_cartesian_spatial_metric.get(0, i, j);
        for (size_t A = 0; A < 2; ++A) {
          d_cartesian_spatial_metric->get(k, i, j) +=
              inverse_cartesian_to_spherical_jacobian.get(k, A + 1) *
              spherical_d_cartesian_spatial_metric.get(A + 1, i, j);
        }
      }
    }
  }
}

void cartesian_shift_and_derivatives_from_modes(
    const gsl::not_null<tnsr::I<DataVector, 3>*> cartesian_shift,
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> d_cartesian_shift,
    const gsl::not_null<tnsr::I<DataVector, 3>*> dt_cartesian_shift,
    const gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const tnsr::I<ComplexModalVector, 3>& shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dr_shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dt_shift_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const size_t l_max) noexcept {
  const size_t size = get<0, 0>(inverse_cartesian_to_spherical_jacobian).size();
  destructive_resize_components(cartesian_shift, size);
  destructive_resize_components(d_cartesian_shift, size);
  destructive_resize_components(dt_cartesian_shift, size);

  destructive_resize_components(interpolation_buffer, size);
  destructive_resize_components(interpolation_modal_buffer, size);
  destructive_resize_components(eth_buffer, size);

  // Allocation
  SphericaliCartesianJ spherical_d_cartesian_shift{size};

  for (size_t i = 0; i < 3; ++i) {
    // copy the modes to a spin-weighted type for interpolation
    get(*interpolation_modal_buffer).data() = shift_coefficients.get(i);
    Spectral::Swsh::inverse_swsh_transform(
        l_max, 1, make_not_null(&get(*interpolation_buffer)),
        get(*interpolation_modal_buffer));
    cartesian_shift->get(i) = real(get(*interpolation_buffer).data());

    get(*interpolation_modal_buffer).data() = dt_shift_coefficients.get(i);
    Spectral::Swsh::inverse_swsh_transform(
        l_max, 1, make_not_null(&get(*interpolation_buffer)),
        get(*interpolation_modal_buffer));
    dt_cartesian_shift->get(i) = real(get(*interpolation_buffer).data());

    get(*interpolation_modal_buffer).data() = dr_shift_coefficients.get(i);
    Spectral::Swsh::inverse_swsh_transform(
        l_max, 1, make_not_null(&get(*interpolation_buffer)),
        get(*interpolation_modal_buffer));
    spherical_d_cartesian_shift.get(0, i) =
        real(get(*interpolation_buffer).data());
  }

  for (size_t i = 0; i < 3; ++i) {
    // reusing the interpolation buffer for taking the angular derivatives
    get(*interpolation_buffer) =
        std::complex<double>(1.0, 0.0) * cartesian_shift->get(i);
    Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
        l_max, 1, make_not_null(&get(*eth_buffer)), get(*interpolation_buffer));
    spherical_d_cartesian_shift.get(1, i) = -real(get(*eth_buffer).data());
    spherical_d_cartesian_shift.get(2, i) = -imag(get(*eth_buffer).data());
  }

  // convert derivatives to cartesian form
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      d_cartesian_shift->get(k, i) =
          inverse_cartesian_to_spherical_jacobian.get(k, 0) *
          spherical_d_cartesian_shift.get(0, i);
      for (size_t A = 0; A < 2; ++A) {
        d_cartesian_shift->get(k, i) +=
            inverse_cartesian_to_spherical_jacobian.get(k, A + 1) *
            spherical_d_cartesian_shift.get(A + 1, i);
      }
    }
  }
}

void cartesian_lapse_and_derivatives_from_modes(
    const gsl::not_null<Scalar<DataVector>*> cartesian_lapse,
    const gsl::not_null<tnsr::i<DataVector, 3>*> d_cartesian_lapse,
    const gsl::not_null<Scalar<DataVector>*> dt_cartesian_lapse,
    const gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const Scalar<ComplexModalVector>& lapse_coefficients,
    const Scalar<ComplexModalVector>& dr_lapse_coefficients,
    const Scalar<ComplexModalVector>& dt_lapse_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const size_t l_max) noexcept {
  const size_t size = get<0, 0>(inverse_cartesian_to_spherical_jacobian).size();
  destructive_resize_components(cartesian_lapse, size);
  destructive_resize_components(d_cartesian_lapse, size);
  destructive_resize_components(dt_cartesian_lapse, size);

  destructive_resize_components(interpolation_buffer, size);
  destructive_resize_components(interpolation_modal_buffer, size);
  destructive_resize_components(eth_buffer, size);

  // Allocation
  tnsr::i<DataVector, 3> spherical_d_cartesian_lapse{size};
  // copy the modes to a spin-weighted type for interpolation
  get(*interpolation_modal_buffer).data() = get(lapse_coefficients);
  Spectral::Swsh::inverse_swsh_transform(
      l_max, 1, make_not_null(&get(*interpolation_buffer)),
      get(*interpolation_modal_buffer));
  get(*cartesian_lapse) = real(get(*interpolation_buffer).data());

  get(*interpolation_modal_buffer).data() = get(dt_lapse_coefficients);
  Spectral::Swsh::inverse_swsh_transform(
      l_max, 1, make_not_null(&get(*interpolation_buffer)),
      get(*interpolation_modal_buffer));
  get(*dt_cartesian_lapse) = real(get(*interpolation_buffer).data());

  get(*interpolation_modal_buffer).data() = get(dr_lapse_coefficients);
  Spectral::Swsh::inverse_swsh_transform(
      l_max, 1, make_not_null(&get(*interpolation_buffer)),
      get(*interpolation_modal_buffer));
  get<0>(spherical_d_cartesian_lapse) = real(get(*interpolation_buffer).data());

  // reusing the interpolation buffer for taking the angular derivatives
  get(*interpolation_buffer) =
      std::complex<double>(1.0, 0.0) * get(*cartesian_lapse);
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, make_not_null(&get(*eth_buffer)), get(*interpolation_buffer));
  spherical_d_cartesian_lapse.get(1) = -real(get(*eth_buffer).data());
  spherical_d_cartesian_lapse.get(2) = -imag(get(*eth_buffer).data());

  // convert derivatives to cartesian form
  for (size_t k = 0; k < 3; ++k) {
    d_cartesian_lapse->get(k) =
        inverse_cartesian_to_spherical_jacobian.get(k, 0) *
        get<0>(spherical_d_cartesian_lapse);
    for (size_t A = 0; A < 2; ++A) {
      d_cartesian_lapse->get(k) +=
          inverse_cartesian_to_spherical_jacobian.get(k, A + 1) *
          spherical_d_cartesian_lapse.get(A + 1);
    }
  }
}

void null_metric_and_derivative(
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::RadialNull>*>
        du_null_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::RadialNull>*>
        null_metric,
    const SphericaliCartesianJ& cartesian_to_spherical_jacobian,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::aa<DataVector, 3>& spacetime_metric) noexcept {
  const size_t size = get<0, 0>(spacetime_metric).size();
  destructive_resize_components(null_metric, size);
  destructive_resize_components(du_null_metric, size);

  get<0, 0>(*null_metric) = get<0, 0>(spacetime_metric);
  get<0, 0>(*du_null_metric) = get<0, 0>(dt_spacetime_metric);

  get<0, 1>(*null_metric) = -1.0;
  get<0, 1>(*du_null_metric) = 0.0;

  for (size_t i = 0; i < 3; ++i) {
    null_metric->get(1, i + 1) = 0.0;
    du_null_metric->get(1, i + 1) = 0.0;
  }

  for (size_t A = 0; A < 2; ++A) {
    null_metric->get(0, A + 2) = cartesian_to_spherical_jacobian.get(A + 1, 0) *
                                 spacetime_metric.get(0, 1);
    du_null_metric->get(0, A + 2) =
        cartesian_to_spherical_jacobian.get(A + 1, 0) *
        dt_spacetime_metric.get(0, 1);
    for (size_t i = 1; i < 3; ++i) {
      null_metric->get(0, A + 2) +=
          cartesian_to_spherical_jacobian.get(A + 1, i) *
          spacetime_metric.get(0, i + 1);
      du_null_metric->get(0, A + 2) +=
          cartesian_to_spherical_jacobian.get(A + 1, i) *
          dt_spacetime_metric.get(0, i + 1);
    }
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      null_metric->get(A + 2, B + 2) =
          cartesian_to_spherical_jacobian.get(A + 1, 0) *
          cartesian_to_spherical_jacobian.get(B + 1, 0) *
          spacetime_metric.get(1, 1);
      du_null_metric->get(A + 2, B + 2) =
          cartesian_to_spherical_jacobian.get(A + 1, 0) *
          cartesian_to_spherical_jacobian.get(B + 1, 0) *
          dt_spacetime_metric.get(1, 1);

      for (size_t i = 1; i < 3; ++i) {
        null_metric->get(A + 2, B + 2) +=
            cartesian_to_spherical_jacobian.get(A + 1, i) *
            cartesian_to_spherical_jacobian.get(B + 1, i) *
            spacetime_metric.get(i + 1, i + 1);
        du_null_metric->get(A + 2, B + 2) +=
            cartesian_to_spherical_jacobian.get(A + 1, i) *
            cartesian_to_spherical_jacobian.get(B + 1, i) *
            dt_spacetime_metric.get(i + 1, i + 1);
      }

      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i + 1; j < 3; ++j) {
          // the off-diagonal pieces must be explicitly symmetrized
          null_metric->get(A + 2, B + 2) +=
              (cartesian_to_spherical_jacobian.get(A + 1, i) *
                   cartesian_to_spherical_jacobian.get(B + 1, j) +
               cartesian_to_spherical_jacobian.get(A + 1, j) *
                   cartesian_to_spherical_jacobian.get(B + 1, i)) *
              spacetime_metric.get(i + 1, j + 1);
          du_null_metric->get(A + 2, B + 2) +=
              (cartesian_to_spherical_jacobian.get(A + 1, i) *
                   cartesian_to_spherical_jacobian.get(B + 1, j) +
               cartesian_to_spherical_jacobian.get(A + 1, j) *
                   cartesian_to_spherical_jacobian.get(B + 1, i)) *
              dt_spacetime_metric.get(i + 1, j + 1);
        }
      }
    }
  }

  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < a; ++b) {
      null_metric->get(a, b) = null_metric->get(b, a);
      du_null_metric->get(a, b) = du_null_metric->get(b, a);
    }
  }
}

void worldtube_normal_and_derivatives(
    const gsl::not_null<tnsr::I<DataVector, 3>*> worldtube_normal,
    const gsl::not_null<tnsr::I<DataVector, 3>*> dt_worldtube_normal,
    const Scalar<DataVector>& cos_phi, const Scalar<DataVector>& cos_theta,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const Scalar<DataVector>& sin_phi, const Scalar<DataVector>& sin_theta,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric) noexcept {
  const size_t size = get<0, 0>(spacetime_metric).size();
  destructive_resize_components(worldtube_normal, size);
  destructive_resize_components(dt_worldtube_normal, size);

  // Allocation
  Variables<tmpl::list<::Tags::Tempi<0, 3>, ::Tags::TempScalar<1>>>
      aggregated_buffers{size};
  tnsr::i<DataVector, 3>& sigma = get<::Tags::Tempi<0, 3>>(aggregated_buffers);
  get<0>(sigma) = get(cos_phi) * square(get(sin_theta));
  get<1>(sigma) = get(sin_phi) * square(get(sin_theta));
  get<2>(sigma) = get(sin_theta) * get(cos_theta);

  // Allocation
  magnitude(make_not_null(&get<::Tags::TempScalar<1>>(aggregated_buffers)),
            sigma, inverse_spatial_metric);
  const DataVector& norm_of_sigma =
      get(get<::Tags::TempScalar<1>>(aggregated_buffers));

  get<0>(sigma) /= norm_of_sigma;
  get<1>(sigma) /= norm_of_sigma;
  get<2>(sigma) /= norm_of_sigma;

  for (size_t i = 0; i < 3; ++i) {
    worldtube_normal->get(i) = inverse_spatial_metric.get(i, 0) * sigma.get(0);
    for (size_t j = 1; j < 3; ++j) {
      worldtube_normal->get(i) +=
          inverse_spatial_metric.get(i, j) * sigma.get(j);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t m = 0; m < 3; ++m) {
      for (size_t n = 0; n < 3; ++n) {
        if (UNLIKELY(m == 0 and n == 0)) {
          dt_worldtube_normal->get(i) =
              (0.5 * worldtube_normal->get(i) * get<0>(*worldtube_normal) -
               inverse_spatial_metric.get(i, 0)) *
              get<0>(*worldtube_normal) * get<1, 1>(dt_spacetime_metric);
        } else {
          dt_worldtube_normal->get(i) +=
              (0.5 * worldtube_normal->get(i) * worldtube_normal->get(m) -
               inverse_spatial_metric.get(i, m)) *
              worldtube_normal->get(n) * dt_spacetime_metric.get(m + 1, n + 1);
        }
      }
    }
  }
}

void null_vector_l_and_derivatives(
    const gsl::not_null<tnsr::A<DataVector, 3>*> du_null_l,
    const gsl::not_null<tnsr::A<DataVector, 3>*> null_l,
    const tnsr::I<DataVector, 3>& dt_worldtube_normal,
    const Scalar<DataVector>& dt_lapse,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::I<DataVector, 3>& dt_shift, const Scalar<DataVector>& lapse,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& worldtube_normal) noexcept {
  const size_t size = get(lapse).size();
  destructive_resize_components(du_null_l, size);
  destructive_resize_components(null_l, size);

  // Allocation
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>>>
      aggregated_buffer{size};
  DataVector& denominator = get(get<::Tags::TempScalar<0>>(aggregated_buffer));
  DataVector& du_denominator =
      get(get<::Tags::TempScalar<1>>(aggregated_buffer));
  DataVector& one_divided_by_lapse =
      get(get<::Tags::TempScalar<2>>(aggregated_buffer));
  one_divided_by_lapse = 1.0 / get(lapse);
  denominator = get(lapse);
  for (size_t i = 0; i < 3; ++i) {
    // off-diagonal
    for (size_t j = i + 1; j < 3; ++j) {
      denominator -= spacetime_metric.get(i + 1, j + 1) *
                     (shift.get(i) * worldtube_normal.get(j) +
                      shift.get(j) * worldtube_normal.get(i));
    }
    // diagonal
    denominator -= spacetime_metric.get(i + 1, i + 1) * shift.get(i) *
                   worldtube_normal.get(i);
  }
  // buffer re-use because we won't need the uninverted denominator after this.
  DataVector& one_divided_by_denominator =
      get(get<::Tags::TempScalar<0>>(aggregated_buffer));
  one_divided_by_denominator = 1.0 / denominator;
  get<0>(*null_l) = one_divided_by_denominator * one_divided_by_lapse;
  for (size_t i = 0; i < 3; ++i) {
    null_l->get(i + 1) =
        (worldtube_normal.get(i) - shift.get(i) * one_divided_by_lapse) *
        one_divided_by_denominator;
  }

  du_denominator = -get(dt_lapse);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i + 1; j < 3; ++j) {
      // symmetry
      du_denominator += (dt_shift.get(i) * worldtube_normal.get(j) +
                         dt_shift.get(j) * worldtube_normal.get(i)) *
                            spacetime_metric.get(i + 1, j + 1) +
                        (shift.get(i) * worldtube_normal.get(j) +
                         shift.get(j) * worldtube_normal.get(i)) *
                            dt_spacetime_metric.get(i + 1, j + 1) +
                        (shift.get(i) * dt_worldtube_normal.get(j) +
                         shift.get(j) * dt_worldtube_normal.get(i)) *
                            spacetime_metric.get(i + 1, j + 1);
    }
    // diagonal
    du_denominator += dt_shift.get(i) * spacetime_metric.get(i + 1, i + 1) *
                          worldtube_normal.get(i) +
                      shift.get(i) * dt_spacetime_metric.get(i + 1, i + 1) *
                          worldtube_normal.get(i) +
                      shift.get(i) * spacetime_metric.get(i + 1, i + 1) *
                          dt_worldtube_normal.get(i);
  }
  du_denominator *= square(one_divided_by_denominator);

  get<0>(*du_null_l) = (du_denominator - get(dt_lapse) * one_divided_by_lapse *
                                             one_divided_by_denominator) *
                       one_divided_by_lapse;
  for (size_t i = 0; i < 3; ++i) {
    du_null_l->get(i + 1) =
        (dt_worldtube_normal.get(i) - dt_shift.get(i) * one_divided_by_lapse) *
        one_divided_by_denominator;
    du_null_l->get(i + 1) += shift.get(i) * get(dt_lapse) *
                             square(one_divided_by_lapse) *
                             one_divided_by_denominator;
    du_null_l->get(i + 1) +=
        (-shift.get(i) * one_divided_by_lapse + worldtube_normal.get(i)) *
        du_denominator;
  }
}

void dlambda_null_metric_and_inverse(
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::RadialNull>*>
        dlambda_null_metric,
    const gsl::not_null<tnsr::AA<DataVector, 3, Frame::RadialNull>*>
        dlambda_inverse_null_metric,
    const AngulariCartesianA& angular_d_null_l,
    const SphericaliCartesianJ& cartesian_to_spherical_jacobian,
    const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::A<DataVector, 3>& du_null_l,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>& inverse_null_metric,
    const tnsr::A<DataVector, 3>& null_l,
    const tnsr::aa<DataVector, 3>& spacetime_metric) noexcept {
  // first, the (down-index) null metric
  const size_t size = get<0, 0>(spacetime_metric).size();
  destructive_resize_components(dlambda_null_metric, size);
  destructive_resize_components(dlambda_inverse_null_metric, size);

  get<0, 0>(*dlambda_null_metric) =
      get<0>(null_l) * get<0, 0>(dt_spacetime_metric) +
      2.0 * get<0>(du_null_l) * get<0, 0>(spacetime_metric);
  for (size_t i = 0; i < 3; ++i) {
    get<0, 0>(*dlambda_null_metric) +=
        null_l.get(i + 1) * phi.get(i, 0, 0) +
        2.0 * du_null_l.get(i + 1) * spacetime_metric.get(i + 1, 0);
  }
  // A0 component
  for (size_t A = 0; A < 2; ++A) {
    dlambda_null_metric->get(0, A + 2) =
        cartesian_to_spherical_jacobian.get(A + 1, 0) *
            (get<0>(du_null_l) * get<1, 0>(spacetime_metric) +
             get<0>(null_l) * get<1, 0>(dt_spacetime_metric)) +
        angular_d_null_l.get(A, 1) * get<1, 0>(spacetime_metric) +
        angular_d_null_l.get(A, 0) * get<0, 0>(spacetime_metric);
    for (size_t k = 1; k < 3; ++k) {
      dlambda_null_metric->get(0, A + 2) +=
          cartesian_to_spherical_jacobian.get(A + 1, k) *
              (get<0>(du_null_l) * spacetime_metric.get(k + 1, 0) +
               get<0>(null_l) * dt_spacetime_metric.get(k + 1, 0)) +
          angular_d_null_l.get(A, k + 1) * spacetime_metric.get(k + 1, 0);
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t k = 0; k < 3; ++k) {
        dlambda_null_metric->get(0, A + 2) +=
            cartesian_to_spherical_jacobian.get(A + 1, k) *
            (du_null_l.get(i + 1) * spacetime_metric.get(k + 1, i + 1) +
             null_l.get(i + 1) * phi.get(i, k + 1, 0));
      }
    }
    dlambda_null_metric->get(A + 2, 0) = dlambda_null_metric->get(0, A + 2);
  }
  // zero the null directions
  get<0, 1>(*dlambda_null_metric) = 0.0;
  for (size_t a = 1; a < 4; ++a) {
    dlambda_null_metric->get(1, a) = 0.0;
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      dlambda_null_metric->get(A + 2, B + 2) = 0.0;
    }
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          dlambda_null_metric->get(A + 2, B + 2) +=
              get<0>(null_l) * cartesian_to_spherical_jacobian.get(A + 1, i) *
              cartesian_to_spherical_jacobian.get(B + 1, j) *
              dt_spacetime_metric.get(i + 1, j + 1);
        }
      }
    }
  }
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          for (size_t k = 0; k < 3; ++k) {
            dlambda_null_metric->get(A + 2, B + 2) +=
                null_l.get(k + 1) *
                cartesian_to_spherical_jacobian.get(A + 1, i) *
                cartesian_to_spherical_jacobian.get(B + 1, j) *
                phi.get(k, i + 1, j + 1);
          }
        }
      }
    }
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t a = 0; a < 4; ++a) {
          dlambda_null_metric->get(A + 2, B + 2) +=
              (angular_d_null_l.get(A, a) *
                   cartesian_to_spherical_jacobian.get(B + 1, i) +
               angular_d_null_l.get(B, a) *
                   cartesian_to_spherical_jacobian.get(A + 1, i)) *
              spacetime_metric.get(a, i + 1);
        }
      }
    }
  }
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < a; ++b) {
      dlambda_null_metric->get(a, b) = dlambda_null_metric->get(b, a);
    }
  }

  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < 4; ++b) {
      dlambda_inverse_null_metric->get(a, b) = 0.0;
    }
  }
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      for (size_t C = 0; C < 2; ++C) {
        for (size_t D = 0; D < 2; ++D) {
          dlambda_inverse_null_metric->get(A + 2, B + 2) -=
              inverse_null_metric.get(A + 2, C + 2) *
              inverse_null_metric.get(B + 2, D + 2) *
              dlambda_null_metric->get(C + 2, D + 2);
        }
      }
    }
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      dlambda_inverse_null_metric->get(1, A + 2) +=
          inverse_null_metric.get(A + 2, B + 2) *
          dlambda_null_metric->get(0, B + 2);
      for (size_t C = 0; C < 2; ++C) {
        dlambda_inverse_null_metric->get(1, A + 2) -=
            inverse_null_metric.get(A + 2, B + 2) *
            inverse_null_metric.get(1, C + 2) *
            dlambda_null_metric->get(C + 2, B + 2);
      }
    }
  }

  get<1, 1>(*dlambda_inverse_null_metric) -= get<0, 0>(*dlambda_null_metric);

  for (size_t A = 0; A < 2; ++A) {
    get<1, 1>(*dlambda_inverse_null_metric) +=
        2.0 * inverse_null_metric.get(1, A + 2) *
        dlambda_null_metric->get(0, A + 2);
    for (size_t B = 0; B < 2; ++B) {
      get<1, 1>(*dlambda_inverse_null_metric) -=
          inverse_null_metric.get(1, A + 2) *
          inverse_null_metric.get(1, B + 2) *
          dlambda_null_metric->get(A + 2, B + 2);
    }
  }
}

void bondi_r(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_r,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& null_metric) noexcept {
  destructive_resize_components(bondi_r, get<0, 0>(null_metric).size());

  // the inclusion of the std::complex<double> informs the expression
  // templates to turn the result into a ComplexDataVector
  get(*bondi_r).data() = std::complex<double>(1.0, 0) *
                         pow(get<2, 2>(null_metric) * get<3, 3>(null_metric) -
                                 square(get<2, 3>(null_metric)),
                             0.25);
}

void d_bondi_r(
    const gsl::not_null<tnsr::a<DataVector, 3, Frame::RadialNull>*> d_bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& dlambda_null_metric,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& du_null_metric,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>& inverse_null_metric,
    const size_t l_max) noexcept {
  destructive_resize_components(d_bondi_r,
                                get<0, 0>(inverse_null_metric).size());

  // compute the time derivative part
  get<0>(*d_bondi_r) =
      0.25 * real(get(bondi_r).data()) *
      (get<2, 2>(inverse_null_metric) * get<2, 2>(du_null_metric) +
       2.0 * get<2, 3>(inverse_null_metric) * get<2, 3>(du_null_metric) +
       get<3, 3>(inverse_null_metric) * get<3, 3>(du_null_metric));
  // compute the lambda derivative part
  get<1>(*d_bondi_r) =
      0.25 * real(get(bondi_r).data()) *
      (get<2, 2>(inverse_null_metric) * get<2, 2>(dlambda_null_metric) +
       2.0 * get<2, 3>(inverse_null_metric) * get<2, 3>(dlambda_null_metric) +
       get<3, 3>(inverse_null_metric) * get<3, 3>(dlambda_null_metric));

  // Allocation (of result and coefficient buffer)
  const auto eth_of_r =
      Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::Eth>(
          l_max, 1, get(bondi_r));
  d_bondi_r->get(2) = -real(eth_of_r.data());
  d_bondi_r->get(3) = -imag(eth_of_r.data());
}

void dyads(
    const gsl::not_null<tnsr::i<ComplexDataVector, 2, Frame::RadialNull>*>
        down_dyad,
    const gsl::not_null<tnsr::I<ComplexDataVector, 2, Frame::RadialNull>*>
        up_dyad) noexcept {
  // implicit factors of sin_theta omitted (still normalized as desired, though)
  get<0>(*down_dyad) = -1.0;
  get<1>(*down_dyad) = std::complex<double>(0.0, -1.0);
  get<0>(*up_dyad) = -1.0;
  get<1>(*up_dyad) = std::complex<double>(0.0, -1.0);
}

void beta_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r) noexcept {
  destructive_resize_components(beta, get<0>(d_bondi_r).size());
  get(*beta).data() = std::complex<double>(-0.5, 0.0) * log(get<1>(d_bondi_r));
}

void bondi_u_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> bondi_u,
    const tnsr::i<ComplexDataVector, 2, Frame::RadialNull>& dyad,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>&
        inverse_null_metric) noexcept {
  destructive_resize_components(bondi_u, get<0>(d_bondi_r).size());
  get(*bondi_u).data() = -get<0>(dyad) * get<1, 2>(inverse_null_metric) -
                         get<1>(dyad) * get<1, 3>(inverse_null_metric);

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      get(*bondi_u).data() -= d_bondi_r.get(2 + A) * dyad.get(B) *
                              inverse_null_metric.get(A + 2, B + 2) /
                              get<1>(d_bondi_r);
    }
  }
}

void bondi_w_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_w,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>& inverse_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r) noexcept {
  destructive_resize_components(bondi_w, get(bondi_r).data().size());

  get(*bondi_w).data() =
      std::complex<double>(1.0, 0.0) *
      (-1.0 + get<1>(d_bondi_r) * get<1, 1>(inverse_null_metric) -
       2.0 * get<0>(d_bondi_r));

  for (size_t A = 0; A < 2; ++A) {
    get(*bondi_w).data() +=
        2.0 * d_bondi_r.get(A + 2) * inverse_null_metric.get(1, A + 2);
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      get(*bondi_w).data() += d_bondi_r.get(A + 2) * d_bondi_r.get(B + 2) *
                              inverse_null_metric.get(A + 2, B + 2) /
                              get<1>(d_bondi_r);
    }
  }
  get(*bondi_w).data() /= get(bondi_r).data();
}

void bondi_j_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> bondi_j,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) noexcept {
  destructive_resize_components(bondi_j, get(bondi_r).data().size());

  get(*bondi_j).data() =
      0.5 *
      (square(get<0>(dyad)) * get<2, 2>(null_metric) +
       2.0 * get<0>(dyad) * get<1>(dyad) * get<2, 3>(null_metric) +
       square(get<1>(dyad)) * get<3, 3>(null_metric)) /
      square(get(bondi_r).data());
}

void dr_bondi_j(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_bondi_j,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        denominator_buffer,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& dlambda_null_metric,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) noexcept {
  destructive_resize_components(dr_bondi_j, get(bondi_r).data().size());
  destructive_resize_components(denominator_buffer, get(bondi_r).data().size());
  get(*dr_bondi_j) = -2.0 * get(bondi_j) / get(bondi_r);
  get(*denominator_buffer).data() =
      1.0 / (square(get(bondi_r).data()) * get<1>(d_bondi_r));
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      get(*dr_bondi_j).data() += 0.5 * dyad.get(A) * dyad.get(B) *
                                 dlambda_null_metric.get(A + 2, B + 2) *
                                 get(*denominator_buffer).data();
    }
  }
}

void d2lambda_bondi_r(
    const gsl::not_null<Scalar<DataVector>*> d2lambda_bondi_r,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r) noexcept {
  destructive_resize_components(d2lambda_bondi_r, get(bondi_j).data().size());
  get(*d2lambda_bondi_r) =
      real(-0.25 * get(bondi_r).data() *
           (get(dr_bondi_j).data() * conj(get(dr_bondi_j).data()) -
            0.25 *
                square(conj(get(bondi_j).data()) * get(dr_bondi_j).data() +
                       get(bondi_j).data() * conj(get(dr_bondi_j).data())) /
                (1.0 + get(bondi_j).data() * conj(get(bondi_j).data()))));
  get(*d2lambda_bondi_r) *= square(get<1>(d_bondi_r));
}

void bondi_q_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> bondi_q,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dr_bondi_u,
    const Scalar<DataVector>& d2lambda_r,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>&
        dlambda_inverse_null_metric,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const tnsr::i<ComplexDataVector, 2, Frame::RadialNull>& dyad,
    const tnsr::i<DataVector, 2, Frame::RadialNull>& angular_d_dlambda_r,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>& inverse_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u) noexcept {
  destructive_resize_components(bondi_q, get(bondi_j).data().size());
  // Allocation
  Scalar<SpinWeighted<ComplexDataVector, 1>> dlambda_bondi_u{
      get(bondi_j).data().size()};

  get(dlambda_bondi_u).data() =
      -get(bondi_u).data() * get(d2lambda_r) / get<1>(d_bondi_r);

  for (size_t A = 0; A < 2; ++A) {
    get(dlambda_bondi_u) -=
        (dlambda_inverse_null_metric.get(1, A + 2) +
         get(d2lambda_r) * inverse_null_metric.get(1, A + 2) /
             get<1>(d_bondi_r)) *
        dyad.get(A);
    for (size_t B = 0; B < 2; ++B) {
      get(dlambda_bondi_u) -=
          (d_bondi_r.get(B + 2) *
           dlambda_inverse_null_metric.get(A + 2, B + 2) / get<1>(d_bondi_r)) *
          dyad.get(A);
      get(dlambda_bondi_u) -= angular_d_dlambda_r.get(B) *
                              inverse_null_metric.get(A + 2, B + 2) *
                              dyad.get(A) / get<1>(d_bondi_r);
    }
  }
  get(*dr_bondi_u).data() = get(dlambda_bondi_u).data() / get<1>(d_bondi_r);

  get(*bondi_q).data() =
      square(get(bondi_r).data()) *
      (get(bondi_j).data() * conj(get(dlambda_bondi_u).data()) +
       sqrt(1.0 + get(bondi_j).data() * conj(get(bondi_j).data())) *
           get(dlambda_bondi_u).data());
}

void bondi_h_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> bondi_h,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& du_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) noexcept {
  destructive_resize_components(bondi_h, get(bondi_j).data().size());

  get(*bondi_h).data() =
      -2.0 * get<0>(d_bondi_r) / get(bondi_r).data() * get(bondi_j).data();
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      get(*bondi_h).data() += (0.5 / square(get(bondi_r).data())) *
                              dyad.get(A) * dyad.get(B) *
                              du_null_metric.get(A + 2, B + 2);
    }
  }
}

void du_j_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> du_bondi_j,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& du_null_metric,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& dlambda_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) noexcept {
  destructive_resize_components(du_bondi_j, get(bondi_j).data().size());

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      if (UNLIKELY(A == 0 and B == 0)) {
        get(*du_bondi_j).data() = -(0.5 / square(get(bondi_r).data())) *
                                  square(get<0>(dyad)) *
                                  (get<0>(d_bondi_r) / get<1>(d_bondi_r) *
                                       get<2, 2>(dlambda_null_metric) -
                                   get<2, 2>(du_null_metric));

      } else {
        get(*du_bondi_j).data() -= (0.5 / square(get(bondi_r).data())) *
                                   dyad.get(A) * dyad.get(B) *
                                   (get<0>(d_bondi_r) / get<1>(d_bondi_r) *
                                        dlambda_null_metric.get(A + 2, B + 2) -
                                    du_null_metric.get(A + 2, B + 2));
      }
    }
  }
}
}  // namespace Cce
