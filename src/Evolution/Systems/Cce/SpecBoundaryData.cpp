// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/BoundaryData.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

void cartesian_spatial_metric_and_derivatives_from_unnormalized_spec_modes(
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
    const gsl::not_null<Scalar<DataVector>*> radial_correction_factor,
    const tnsr::ii<ComplexModalVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dt_spatial_metric_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const tnsr::I<DataVector, 3>& unit_cartesian_coords, const size_t l_max) {
  const size_t size = get<0, 0>(inverse_cartesian_to_spherical_jacobian).size();
  destructive_resize_components(cartesian_spatial_metric, size);
  destructive_resize_components(d_cartesian_spatial_metric, size);
  destructive_resize_components(dt_cartesian_spatial_metric, size);

  destructive_resize_components(interpolation_buffer, size);
  destructive_resize_components(interpolation_modal_buffer, size);
  destructive_resize_components(eth_buffer, size);
  destructive_resize_components(radial_correction_factor, size);

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

  get(*radial_correction_factor) = square(get<0>(unit_cartesian_coords)) *
                                   get<0, 0>(*inverse_cartesian_spatial_metric);
  for (size_t i = 1; i < 3; ++i) {
    get(*radial_correction_factor) +=
        square(unit_cartesian_coords.get(i)) *
        inverse_cartesian_spatial_metric->get(i, i);
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i + 1; j < 3; ++j) {
      get(*radial_correction_factor) +=
          2.0 * inverse_cartesian_spatial_metric->get(i, j) *
          unit_cartesian_coords.get(i) * unit_cartesian_coords.get(j);
    }
  }
  get(*radial_correction_factor) = sqrt(get(*radial_correction_factor));
  // convert derivatives to cartesian form
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        d_cartesian_spatial_metric->get(k, i, j) =
            inverse_cartesian_to_spherical_jacobian.get(k, 0) *
            spherical_d_cartesian_spatial_metric.get(0, i, j) *
            get(*radial_correction_factor);
        for (size_t A = 0; A < 2; ++A) {
          d_cartesian_spatial_metric->get(k, i, j) +=
              inverse_cartesian_to_spherical_jacobian.get(k, A + 1) *
              spherical_d_cartesian_spatial_metric.get(A + 1, i, j);
        }
      }
    }
  }
}

void cartesian_shift_and_derivatives_from_unnormalized_spec_modes(
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
    const Scalar<DataVector>& radial_derivative_correction_factor,
    const size_t l_max) {
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
          spherical_d_cartesian_shift.get(0, i) *
          get(radial_derivative_correction_factor);
      for (size_t A = 0; A < 2; ++A) {
        d_cartesian_shift->get(k, i) +=
            inverse_cartesian_to_spherical_jacobian.get(k, A + 1) *
            spherical_d_cartesian_shift.get(A + 1, i);
      }
    }
  }
}

void cartesian_lapse_and_derivatives_from_unnormalized_spec_modes(
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
    const Scalar<DataVector>& radial_derivative_correction_factor,
    const size_t l_max) {
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
        get<0>(spherical_d_cartesian_lapse) *
        get(radial_derivative_correction_factor);
    for (size_t A = 0; A < 2; ++A) {
      d_cartesian_lapse->get(k) +=
          inverse_cartesian_to_spherical_jacobian.get(k, A + 1) *
          spherical_d_cartesian_lapse.get(A + 1);
    }
  }
}

}  // namespace Cce
