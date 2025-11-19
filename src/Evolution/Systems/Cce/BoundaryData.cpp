// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/BoundaryData.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"
#include "Evolution/Systems/Cce/SpecBoundaryData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/ComplexDataView.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpacetimeMetric.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

void trigonometric_functions_on_swsh_collocation(
    const gsl::not_null<Scalar<DataVector>*> cos_phi,
    const gsl::not_null<Scalar<DataVector>*> cos_theta,
    const gsl::not_null<Scalar<DataVector>*> sin_phi,
    const gsl::not_null<Scalar<DataVector>*> sin_theta, const size_t l_max) {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  set_number_of_grid_points(cos_phi, size);
  set_number_of_grid_points(cos_theta, size);
  set_number_of_grid_points(sin_phi, size);
  set_number_of_grid_points(sin_theta, size);

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
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
    const double extraction_radius) {
  const size_t size = get(cos_phi).size();
  set_number_of_grid_points(unit_cartesian_coords, size);
  set_number_of_grid_points(cartesian_to_spherical_jacobian, size);
  set_number_of_grid_points(inverse_cartesian_to_spherical_jacobian, size);

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
    const size_t l_max) {
  const size_t size = get<0, 0>(inverse_cartesian_to_spherical_jacobian).size();
  set_number_of_grid_points(cartesian_spatial_metric, size);
  set_number_of_grid_points(d_cartesian_spatial_metric, size);
  set_number_of_grid_points(dt_cartesian_spatial_metric, size);

  set_number_of_grid_points(interpolation_buffer, size);
  set_number_of_grid_points(interpolation_modal_buffer, size);
  set_number_of_grid_points(eth_buffer, size);

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
    const size_t l_max) {
  const size_t size = get<0, 0>(inverse_cartesian_to_spherical_jacobian).size();
  set_number_of_grid_points(cartesian_shift, size);
  set_number_of_grid_points(d_cartesian_shift, size);
  set_number_of_grid_points(dt_cartesian_shift, size);

  set_number_of_grid_points(interpolation_buffer, size);
  set_number_of_grid_points(interpolation_modal_buffer, size);
  set_number_of_grid_points(eth_buffer, size);

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
    const size_t l_max) {
  const size_t size = get<0, 0>(inverse_cartesian_to_spherical_jacobian).size();
  set_number_of_grid_points(cartesian_lapse, size);
  set_number_of_grid_points(d_cartesian_lapse, size);
  set_number_of_grid_points(dt_cartesian_lapse, size);

  set_number_of_grid_points(interpolation_buffer, size);
  set_number_of_grid_points(interpolation_modal_buffer, size);
  set_number_of_grid_points(eth_buffer, size);

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

void deriv_cartesian_metric_lapse_shift_from_nodes(
    const gsl::not_null<tnsr::ijj<DataVector, 3>*> d_cartesian_spatial_metric,
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> d_cartesian_shift,
    const gsl::not_null<tnsr::i<DataVector, 3>*> d_cartesian_lapse,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> buffer,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const tnsr::ii<DataVector, 3>& cartesian_spatial_metric,
    const tnsr::ii<DataVector, 3>& dr_cartesian_spatial_metric,
    const tnsr::I<DataVector, 3>& cartesian_shift,
    const tnsr::I<DataVector, 3>& dr_cartesian_shift,
    const Scalar<DataVector>& cartesian_lapse,
    const Scalar<DataVector>& dr_cartesian_lapse,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const size_t l_max) {
  const size_t size = get<0, 0>(inverse_cartesian_to_spherical_jacobian).size();
  set_number_of_grid_points(buffer, size);
  set_number_of_grid_points(eth_buffer, size);

  set_number_of_grid_points(d_cartesian_spatial_metric, size);
  set_number_of_grid_points(d_cartesian_shift, size);
  set_number_of_grid_points(d_cartesian_lapse, size);

  // Allocations
  SphericaliCartesianjj spherical_d_cartesian_spatial_metric{size};
  SphericaliCartesianJ spherical_d_cartesian_shift{size};
  tnsr::i<DataVector, 3> spherical_d_cartesian_lapse{size};

  // Radial derivative is just a copy
  get<0>(spherical_d_cartesian_lapse) = get(dr_cartesian_lapse);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      spherical_d_cartesian_spatial_metric.get(0, i, j) =
          dr_cartesian_spatial_metric.get(i, j);
    }
    spherical_d_cartesian_shift.get(0, i) = dr_cartesian_shift.get(i);
  }

  // Compute angular derivatives
  get(*buffer) = std::complex<double>(1.0, 0.0) * get(cartesian_lapse);
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, make_not_null(&get(*eth_buffer)), get(*buffer));
  get<1>(spherical_d_cartesian_lapse) = -real(get(*eth_buffer).data());
  get<2>(spherical_d_cartesian_lapse) = -imag(get(*eth_buffer).data());
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      get(*buffer) =
          std::complex<double>(1.0, 0.0) * cartesian_spatial_metric.get(i, j);
      Spectral::Swsh::angular_derivatives<
          tmpl::list<Spectral::Swsh::Tags::Eth>>(
          l_max, 1, make_not_null(&get(*eth_buffer)), get(*buffer));
      spherical_d_cartesian_spatial_metric.get(1, i, j) =
          -real(get(*eth_buffer).data());
      spherical_d_cartesian_spatial_metric.get(2, i, j) =
          -imag(get(*eth_buffer).data());
    }

    get(*buffer) = std::complex<double>(1.0, 0.0) * cartesian_shift.get(i);
    Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
        l_max, 1, make_not_null(&get(*eth_buffer)), get(*buffer));
    spherical_d_cartesian_shift.get(1, i) = -real(get(*eth_buffer).data());
    spherical_d_cartesian_shift.get(2, i) = -imag(get(*eth_buffer).data());
  }

  // Convert derivatives to cartesian form
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        d_cartesian_spatial_metric->get(k, i, j) =
            inverse_cartesian_to_spherical_jacobian.get(k, 0) *
            spherical_d_cartesian_spatial_metric.get(0, i, j);
        for (size_t A = 0; A < 2; ++A) {
          d_cartesian_spatial_metric->get(k, i, j) +=
              inverse_cartesian_to_spherical_jacobian.get(k, A + 1) *
              spherical_d_cartesian_spatial_metric.get(A + 1, i, j);
        }
      }

      d_cartesian_shift->get(k, i) =
          inverse_cartesian_to_spherical_jacobian.get(k, 0) *
          spherical_d_cartesian_shift.get(0, i);
      for (size_t A = 0; A < 2; ++A) {
        d_cartesian_shift->get(k, i) +=
            inverse_cartesian_to_spherical_jacobian.get(k, A + 1) *
            spherical_d_cartesian_shift.get(A + 1, i);
      }
    }

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
    const tnsr::aa<DataVector, 3>& spacetime_metric) {
  const size_t size = get<0, 0>(spacetime_metric).size();
  set_number_of_grid_points(null_metric, size);
  set_number_of_grid_points(du_null_metric, size);

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
    const tnsr::II<DataVector, 3>& inverse_spatial_metric) {
  const size_t size = get<0, 0>(spacetime_metric).size();

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
    const tnsr::I<DataVector, 3>& worldtube_normal) {
  const size_t size = get(lapse).size();
  CAPTURE_FOR_ERROR(lapse);
  CAPTURE_FOR_ERROR(dt_lapse);
  CAPTURE_FOR_ERROR(shift);
  CAPTURE_FOR_ERROR(dt_shift);

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
  CAPTURE_FOR_ERROR(denominator);
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
    const tnsr::aa<DataVector, 3>& spacetime_metric) {
  // first, the (down-index) null metric
  const size_t size = get<0, 0>(spacetime_metric).size();
  set_number_of_grid_points(dlambda_null_metric, size);
  set_number_of_grid_points(dlambda_inverse_null_metric, size);

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
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& null_metric) {
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
    const size_t l_max) {
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
        up_dyad) {
  // implicit factors of sin_theta omitted (still normalized as desired, though)
  get<0>(*down_dyad) = -1.0;
  get<1>(*down_dyad) = std::complex<double>(0.0, -1.0);
  get<0>(*up_dyad) = -1.0;
  get<1>(*up_dyad) = std::complex<double>(0.0, -1.0);
}

void beta_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r) {
  get(*beta).data() = std::complex<double>(-0.5, 0.0) * log(get<1>(d_bondi_r));
}

void bondi_u_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> bondi_u,
    const tnsr::i<ComplexDataVector, 2, Frame::RadialNull>& dyad,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>& inverse_null_metric) {
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
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r) {
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
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) {
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
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) {
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
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r) {
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
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u) {
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
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) {
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
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*bondi_j*/,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& du_null_metric,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& dlambda_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) {
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

void klein_gordon_psi_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> kg_psi,
    const Scalar<DataVector>& csw_psi) {
  get(*kg_psi).data() = std::complex<double>(1.0, 0.0) * get(csw_psi);
}

void klein_gordon_pi_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> kg_pi,
    const Scalar<DataVector>& csw_pi, const tnsr::i<DataVector, 3>& csw_phi,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift) {
  // Pure time derivative
  // dt Psi = - lapse * Pi + shift^{i} Phi_{i}
  get(*kg_pi).data() =
      std::complex<double>(-1.0, 0.0) * get(lapse) * get(csw_pi);
  for (size_t i = 0; i < 3; i++) {
    get(*kg_pi).data() +=
        std::complex<double>(1.0, 0.0) * shift.get(i) * csw_phi.get(i);
  }
}

namespace {
// the common step between the modal input and the Generalized harmonic input
// that performs the final gauge processing to Bondi scalars and places them in
// the Variables.
template <typename BufferTagList, typename ComplexBufferTagList>
void create_bondi_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        bondi_boundary_data,
    const gsl::not_null<Variables<BufferTagList>*> computation_variables,
    const gsl::not_null<Variables<ComplexBufferTagList>*> derivative_buffers,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::A<DataVector, 3>& null_l,
    const tnsr::A<DataVector, 3>& du_null_l,
    const SphericaliCartesianJ& cartesian_to_spherical_jacobian,
    const size_t l_max, const double extraction_radius) {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // unfortunately, because the dyads are not themselves spin-weighted, they
  // need a separate Variables
  Variables<tmpl::list<Tags::detail::DownDyad, Tags::detail::UpDyad>>
      dyad_variables{size};

  auto& null_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>(
          *computation_variables);
  auto& du_null_metric = get<
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>>(
      *computation_variables);
  null_metric_and_derivative(
      make_not_null(&du_null_metric), make_not_null(&null_metric),
      cartesian_to_spherical_jacobian, dt_spacetime_metric, spacetime_metric);

  auto& inverse_null_metric =
      get<gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>>(
          *computation_variables);

  // the below scaling process is used to reduce accumulation of numerical
  // error in the determinant evaluation

  // buffer reuse because the scaled null metric is only needed until the
  // `determinant_and_inverse` call
  auto& scaled_null_metric =
      get<gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>>(
          *computation_variables);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = i; j < 4; ++j) {
      if (i > 1 and j > 1) {
        scaled_null_metric.get(i, j) =
            null_metric.get(i, j) / square(extraction_radius);
      } else if (i > 1 or j > 1) {
        scaled_null_metric.get(i, j) =
            null_metric.get(i, j) / extraction_radius;
      } else {
        scaled_null_metric.get(i, j) = null_metric.get(i, j);
      }
    }
  }
  // Allocation
  const auto scaled_inverse_null_metric =
      determinant_and_inverse(scaled_null_metric).second;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = i; j < 4; ++j) {
      if (i > 1 and j > 1) {
        inverse_null_metric.get(i, j) =
            scaled_inverse_null_metric.get(i, j) / square(extraction_radius);
      } else if (i > 1 or j > 1) {
        inverse_null_metric.get(i, j) =
            scaled_inverse_null_metric.get(i, j) / extraction_radius;
      } else {
        inverse_null_metric.get(i, j) = scaled_inverse_null_metric.get(i, j);
      }
    }
  }

  auto& angular_d_null_l =
      get<Tags::detail::AngularDNullL>(*computation_variables);
  auto& buffer_for_derivatives =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          *derivative_buffers));
  auto& eth_buffer =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          *derivative_buffers));
  for (size_t a = 0; a < 4; ++a) {
    buffer_for_derivatives.data() =
        std::complex<double>(1.0, 0.0) * null_l.get(a);
    Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
        l_max, 1, make_not_null(&eth_buffer), buffer_for_derivatives);
    angular_d_null_l.get(0, a) = -real(eth_buffer.data());
    angular_d_null_l.get(1, a) = -imag(eth_buffer.data());
  }

  auto& dlambda_null_metric = get<Tags::detail::DLambda<
      gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>>(
      *computation_variables);
  auto& dlambda_inverse_null_metric = get<Tags::detail::DLambda<
      gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>>>(
      *computation_variables);
  dlambda_null_metric_and_inverse(
      make_not_null(&dlambda_null_metric),
      make_not_null(&dlambda_inverse_null_metric), angular_d_null_l,
      cartesian_to_spherical_jacobian, phi, dt_spacetime_metric, du_null_l,
      inverse_null_metric, null_l, spacetime_metric);

  auto& r = get<Tags::BoundaryValue<Tags::BondiR>>(*bondi_boundary_data);
  bondi_r(make_not_null(&r), null_metric);

  auto& d_r =
      get<::Tags::spacetime_deriv<Tags::detail::RealBondiR, tmpl::size_t<3>,
                                  Frame::RadialNull>>(*computation_variables);
  d_bondi_r(make_not_null(&d_r), r, dlambda_null_metric, du_null_metric,
            inverse_null_metric, l_max);
  get(get<Tags::BoundaryValue<Tags::DuRDividedByR>>(*bondi_boundary_data))
      .data() = std::complex<double>(1.0, 0.0) * get<0>(d_r) / get(r).data();
  get(get<Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>(*bondi_boundary_data))
      .data() = std::complex<double>(1.0, 0.0) * get<0>(d_r);

  auto& down_dyad = get<Tags::detail::DownDyad>(dyad_variables);
  auto& up_dyad = get<Tags::detail::UpDyad>(dyad_variables);
  dyads(make_not_null(&down_dyad), make_not_null(&up_dyad));

  beta_worldtube_data(make_not_null(&get<Tags::BoundaryValue<Tags::BondiBeta>>(
                          *bondi_boundary_data)),
                      d_r);

  auto& bondi_u = get<Tags::BoundaryValue<Tags::BondiU>>(*bondi_boundary_data);
  bondi_u_worldtube_data(make_not_null(&bondi_u), down_dyad, d_r,
                         inverse_null_metric);

  bondi_w_worldtube_data(make_not_null(&get<Tags::BoundaryValue<Tags::BondiW>>(
                             *bondi_boundary_data)),
                         d_r, inverse_null_metric, r);

  auto& bondi_j = get<Tags::BoundaryValue<Tags::BondiJ>>(*bondi_boundary_data);
  bondi_j_worldtube_data(make_not_null(&bondi_j), null_metric, r, up_dyad);

  auto& dr_j =
      get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(*bondi_boundary_data);
  auto& denominator_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          *derivative_buffers);
  dr_bondi_j(make_not_null(&get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
                 *bondi_boundary_data)),
             make_not_null(&denominator_buffer), dlambda_null_metric, d_r,
             bondi_j, r, up_dyad);

  auto& d2lambda_r = get<
      Tags::detail::DLambda<Tags::detail::DLambda<Tags::detail::RealBondiR>>>(
      *computation_variables);
  d2lambda_bondi_r(make_not_null(&d2lambda_r), d_r, dr_j, bondi_j, r);

  auto& angular_d_dlambda_r =
      get<::Tags::deriv<Tags::detail::DLambda<Tags::detail::RealBondiR>,
                        tmpl::size_t<2>, Frame::RadialNull>>(
          *computation_variables);
  buffer_for_derivatives.data() = std::complex<double>(1.0, 0.0) * get<1>(d_r);
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, make_not_null(&eth_buffer), buffer_for_derivatives);
  angular_d_dlambda_r.get(0) = -real(eth_buffer.data());
  angular_d_dlambda_r.get(1) = -imag(eth_buffer.data());

  bondi_q_worldtube_data(
      make_not_null(
          &get<Tags::BoundaryValue<Tags::BondiQ>>(*bondi_boundary_data)),
      make_not_null(&get<Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>>(
          *bondi_boundary_data)),
      d2lambda_r, dlambda_inverse_null_metric, d_r, down_dyad,
      angular_d_dlambda_r, inverse_null_metric, bondi_j, r, bondi_u);

  bondi_h_worldtube_data(make_not_null(&get<Tags::BoundaryValue<Tags::BondiH>>(
                             *bondi_boundary_data)),
                         d_r, bondi_j, du_null_metric, r, up_dyad);

  du_j_worldtube_data(
      make_not_null(&get<Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>>(
          *bondi_boundary_data)),
      d_r, bondi_j, du_null_metric, dlambda_null_metric, r, up_dyad);
}
}  // namespace

void create_bondi_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        bondi_boundary_data,
    const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& pi,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const double extraction_radius, const size_t l_max) {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  // Most allocations required for the full boundary computation are merged into
  // a single, large Variables allocation. There remain a handful of cases in
  // the computational functions called where an intermediate quantity that is
  // not re-used is allocated rather than taking a buffer. These cases are
  // marked with code comments 'Allocation'; In the future, if allocations are
  // identified as a point to optimize, those buffers may be allocated here and
  // passed as function arguments
  Variables<tmpl::list<
      Tags::detail::CosPhi, Tags::detail::CosTheta, Tags::detail::SinPhi,
      Tags::detail::SinTheta, Tags::detail::CartesianCoordinates,
      Tags::detail::CartesianToSphericalJacobian,
      Tags::detail::InverseCartesianToSphericalJacobian,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::Shift<DataVector, 3>,
      ::Tags::dt<gr::Tags::Shift<DataVector, 3>>, gr::Tags::Lapse<DataVector>,
      ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>,
      Tags::detail::WorldtubeNormal, ::Tags::dt<Tags::detail::WorldtubeNormal>,
      gr::Tags::SpacetimeNormalVector<DataVector, 3>, Tags::detail::NullL,
      ::Tags::dt<Tags::detail::NullL>,
      // for the detail function called at the end
      gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>,
      Tags::detail::AngularDNullL,
      Tags::detail::DLambda<
          gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      Tags::detail::DLambda<
          gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      ::Tags::spacetime_deriv<Tags::detail::RealBondiR, tmpl::size_t<3>,
                              Frame::RadialNull>,
      Tags::detail::DLambda<Tags::detail::DLambda<Tags::detail::RealBondiR>>,
      ::Tags::deriv<Tags::detail::DLambda<Tags::detail::RealBondiR>,
                    tmpl::size_t<2>, Frame::RadialNull>>>
      computation_variables{size};

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>>
      derivative_buffers{size};

  auto& cos_phi = get<Tags::detail::CosPhi>(computation_variables);
  auto& cos_theta = get<Tags::detail::CosTheta>(computation_variables);
  auto& sin_phi = get<Tags::detail::SinPhi>(computation_variables);
  auto& sin_theta = get<Tags::detail::SinTheta>(computation_variables);
  trigonometric_functions_on_swsh_collocation(
      make_not_null(&cos_phi), make_not_null(&cos_theta),
      make_not_null(&sin_phi), make_not_null(&sin_theta), l_max);

  // NOTE: to handle the singular values of polar coordinates, the phi
  // components of all tensors are scaled according to their sin(theta)
  // prefactors.
  // so, any down-index component get<2>(A) represents 1/sin(theta) A_\phi,
  // and any up-index component get<2>(A) represents sin(theta) A^\phi.
  // This holds for Jacobians, and so direct application of the Jacobians
  // brings the factors through.
  auto& cartesian_coords =
      get<Tags::detail::CartesianCoordinates>(computation_variables);
  auto& cartesian_to_spherical_jacobian =
      get<Tags::detail::CartesianToSphericalJacobian>(computation_variables);
  auto& inverse_cartesian_to_spherical_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          computation_variables);
  cartesian_to_spherical_coordinates_and_jacobians(
      make_not_null(&cartesian_coords),
      make_not_null(&cartesian_to_spherical_jacobian),
      make_not_null(&inverse_cartesian_to_spherical_jacobian), cos_phi,
      cos_theta, sin_phi, sin_theta, extraction_radius);

  auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(computation_variables);
  gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);

  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(computation_variables);
  // Allocation
  inverse_spatial_metric = determinant_and_inverse(spatial_metric).second;

  auto& shift = get<gr::Tags::Shift<DataVector, 3>>(computation_variables);
  gr::shift(make_not_null(&shift), spacetime_metric, inverse_spatial_metric);

  auto& lapse = get<gr::Tags::Lapse<DataVector>>(computation_variables);
  gr::lapse(make_not_null(&lapse), shift, spacetime_metric);

  auto& dt_spacetime_metric =
      get<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
          computation_variables);

  gh::time_derivative_of_spacetime_metric(make_not_null(&dt_spacetime_metric),
                                          lapse, shift, pi, phi);

  auto& dt_worldtube_normal =
      get<::Tags::dt<Tags::detail::WorldtubeNormal>>(computation_variables);
  auto& worldtube_normal =
      get<Tags::detail::WorldtubeNormal>(computation_variables);
  worldtube_normal_and_derivatives(
      make_not_null(&worldtube_normal), make_not_null(&dt_worldtube_normal),
      cos_phi, cos_theta, spacetime_metric, dt_spacetime_metric, sin_phi,
      sin_theta, inverse_spatial_metric);
  auto& spacetime_unit_normal =
      get<gr::Tags::SpacetimeNormalVector<DataVector, 3>>(
          computation_variables);
  gr::spacetime_normal_vector(make_not_null(&spacetime_unit_normal), lapse,
                              shift);
  auto& dt_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(computation_variables);
  gh::time_deriv_of_lapse(make_not_null(&dt_lapse), lapse, shift,
                          spacetime_unit_normal, phi, pi);
  auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(computation_variables);
  gh::time_deriv_of_shift(make_not_null(&dt_shift), lapse, shift,
                          inverse_spatial_metric, spacetime_unit_normal, phi,
                          pi);

  auto& du_null_l = get<::Tags::dt<Tags::detail::NullL>>(computation_variables);
  auto& null_l = get<Tags::detail::NullL>(computation_variables);
  null_vector_l_and_derivatives(make_not_null(&du_null_l),
                                make_not_null(&null_l), dt_worldtube_normal,
                                dt_lapse, dt_spacetime_metric, dt_shift, lapse,
                                spacetime_metric, shift, worldtube_normal);

  // pass to the next step that is common between the 'modal' input and 'GH'
  // input strategies
  create_bondi_boundary_data(
      bondi_boundary_data, make_not_null(&computation_variables),
      make_not_null(&derivative_buffers), dt_spacetime_metric, phi,
      spacetime_metric, null_l, du_null_l, cartesian_to_spherical_jacobian,
      l_max, extraction_radius);
}

void create_bondi_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        bondi_boundary_data,
    const tnsr::ii<ComplexModalVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dt_spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::I<ComplexModalVector, 3>& shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dt_shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dr_shift_coefficients,
    const Scalar<ComplexModalVector>& lapse_coefficients,
    const Scalar<ComplexModalVector>& dt_lapse_coefficients,
    const Scalar<ComplexModalVector>& dr_lapse_coefficients,
    const double extraction_radius, const size_t l_max) {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // Most allocations required for the full boundary computation are merged into
  // a single, large Variables allocation. There remain a handful of cases in
  // the computational functions called where an intermediate quantity that is
  // not re-used is allocated rather than taking a buffer. These cases are
  // marked with code comments 'Allocation'; In the future, if allocations are
  // identified as a point to optimize, those buffers may be allocated here and
  // passed as function arguments
  Variables<tmpl::list<
      Tags::detail::CosPhi, Tags::detail::CosTheta, Tags::detail::SinPhi,
      Tags::detail::SinTheta, Tags::detail::CartesianCoordinates,
      Tags::detail::CartesianToSphericalJacobian,
      Tags::detail::InverseCartesianToSphericalJacobian,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>,
      gr::Tags::Shift<DataVector, 3>,
      ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::Shift<DataVector, 3>>, gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      gr::Tags::SpacetimeMetric<DataVector, 3>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>,
      gh::Tags::Phi<DataVector, 3>, Tags::detail::WorldtubeNormal,
      ::Tags::dt<Tags::detail::WorldtubeNormal>, Tags::detail::NullL,
      ::Tags::dt<Tags::detail::NullL>,
      // for the detail function called at the end
      gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>,
      Tags::detail::AngularDNullL,
      Tags::detail::DLambda<
          gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      Tags::detail::DLambda<
          gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      ::Tags::spacetime_deriv<Tags::detail::RealBondiR, tmpl::size_t<3>,
                              Frame::RadialNull>,
      Tags::detail::DLambda<Tags::detail::DLambda<Tags::detail::RealBondiR>>,
      ::Tags::deriv<Tags::detail::DLambda<Tags::detail::RealBondiR>,
                    tmpl::size_t<2>, Frame::RadialNull>>>
      computation_variables{size};

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>>
      derivative_buffers{size};
  auto& cos_phi = get<Tags::detail::CosPhi>(computation_variables);
  auto& cos_theta = get<Tags::detail::CosTheta>(computation_variables);
  auto& sin_phi = get<Tags::detail::SinPhi>(computation_variables);
  auto& sin_theta = get<Tags::detail::SinTheta>(computation_variables);
  trigonometric_functions_on_swsh_collocation(
      make_not_null(&cos_phi), make_not_null(&cos_theta),
      make_not_null(&sin_phi), make_not_null(&sin_theta), l_max);

  // NOTE: to handle the singular values of polar coordinates, the phi
  // components of all tensors are scaled according to their sin(theta)
  // prefactors.
  // so, any down-index component get<2>(A) represents 1/sin(theta) A_\phi,
  // and any up-index component get<2>(A) represents sin(theta) A^\phi.
  // This holds for Jacobians, and so direct application of the Jacobians
  // brings the factors through.
  auto& cartesian_coords =
      get<Tags::detail::CartesianCoordinates>(computation_variables);
  auto& cartesian_to_spherical_jacobian =
      get<Tags::detail::CartesianToSphericalJacobian>(computation_variables);
  auto& inverse_cartesian_to_spherical_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          computation_variables);
  cartesian_to_spherical_coordinates_and_jacobians(
      make_not_null(&cartesian_coords),
      make_not_null(&cartesian_to_spherical_jacobian),
      make_not_null(&inverse_cartesian_to_spherical_jacobian), cos_phi,
      cos_theta, sin_phi, sin_theta, extraction_radius);

  auto& cartesian_spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(computation_variables);
  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(computation_variables);
  auto& d_cartesian_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);
  auto& dt_cartesian_spatial_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(
          computation_variables);
  auto& interpolation_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          derivative_buffers);
  Scalar<SpinWeighted<ComplexModalVector, 0>> interpolation_modal_buffer{size};
  auto& eth_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 1>>>(
          derivative_buffers);
  cartesian_spatial_metric_and_derivatives_from_modes(
      make_not_null(&cartesian_spatial_metric),
      make_not_null(&inverse_spatial_metric),
      make_not_null(&d_cartesian_spatial_metric),
      make_not_null(&dt_cartesian_spatial_metric),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      spatial_metric_coefficients, dr_spatial_metric_coefficients,
      dt_spatial_metric_coefficients, inverse_cartesian_to_spherical_jacobian,
      l_max);

  auto& cartesian_shift =
      get<gr::Tags::Shift<DataVector, 3>>(computation_variables);
  auto& d_cartesian_shift =
      get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);
  auto& dt_cartesian_shift =
      get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(computation_variables);

  cartesian_shift_and_derivatives_from_modes(
      make_not_null(&cartesian_shift), make_not_null(&d_cartesian_shift),
      make_not_null(&dt_cartesian_shift),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      shift_coefficients, dr_shift_coefficients, dt_shift_coefficients,
      inverse_cartesian_to_spherical_jacobian, l_max);

  auto& cartesian_lapse =
      get<gr::Tags::Lapse<DataVector>>(computation_variables);
  auto& d_cartesian_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);
  auto& dt_cartesian_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(computation_variables);
  cartesian_lapse_and_derivatives_from_modes(
      make_not_null(&cartesian_lapse), make_not_null(&d_cartesian_lapse),
      make_not_null(&dt_cartesian_lapse),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      lapse_coefficients, dr_lapse_coefficients, dt_lapse_coefficients,
      inverse_cartesian_to_spherical_jacobian, l_max);

  auto& phi = get<gh::Tags::Phi<DataVector, 3>>(computation_variables);
  auto& dt_spacetime_metric =
      get<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
          computation_variables);
  auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(computation_variables);
  gh::phi(make_not_null(&phi), cartesian_lapse, d_cartesian_lapse,
          cartesian_shift, d_cartesian_shift, cartesian_spatial_metric,
          d_cartesian_spatial_metric);
  gr::time_derivative_of_spacetime_metric(
      make_not_null(&dt_spacetime_metric), cartesian_lapse, dt_cartesian_lapse,
      cartesian_shift, dt_cartesian_shift, cartesian_spatial_metric,
      dt_cartesian_spatial_metric);
  gr::spacetime_metric(make_not_null(&spacetime_metric), cartesian_lapse,
                       cartesian_shift, cartesian_spatial_metric);

  auto& dt_worldtube_normal =
      get<::Tags::dt<Tags::detail::WorldtubeNormal>>(computation_variables);
  auto& worldtube_normal =
      get<Tags::detail::WorldtubeNormal>(computation_variables);
  worldtube_normal_and_derivatives(
      make_not_null(&worldtube_normal), make_not_null(&dt_worldtube_normal),
      cos_phi, cos_theta, spacetime_metric, dt_spacetime_metric, sin_phi,
      sin_theta, inverse_spatial_metric);

  auto& du_null_l = get<::Tags::dt<Tags::detail::NullL>>(computation_variables);
  auto& null_l = get<Tags::detail::NullL>(computation_variables);
  null_vector_l_and_derivatives(
      make_not_null(&du_null_l), make_not_null(&null_l), dt_worldtube_normal,
      dt_cartesian_lapse, dt_spacetime_metric, dt_cartesian_shift,
      cartesian_lapse, spacetime_metric, cartesian_shift, worldtube_normal);

  // pass to the next step that is common between the 'modal' input and 'GH'
  // input strategies
  create_bondi_boundary_data(
      bondi_boundary_data, make_not_null(&computation_variables),
      make_not_null(&derivative_buffers), dt_spacetime_metric, phi,
      spacetime_metric, null_l, du_null_l, cartesian_to_spherical_jacobian,
      l_max, extraction_radius);
}

void create_bondi_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        bondi_boundary_data,
    const tnsr::ii<DataVector, 3>& cartesian_spatial_metric,
    const tnsr::ii<DataVector, 3>& cartesian_dt_spatial_metric,
    const tnsr::ii<DataVector, 3>& cartesian_dr_spatial_metric,
    const tnsr::I<DataVector, 3>& cartesian_shift,
    const tnsr::I<DataVector, 3>& cartesian_dt_shift,
    const tnsr::I<DataVector, 3>& cartesian_dr_shift,
    const Scalar<DataVector>& cartesian_lapse,
    const Scalar<DataVector>& cartesian_dt_lapse,
    const Scalar<DataVector>& cartesian_dr_lapse,
    const double extraction_radius, const size_t l_max) {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Variables<tmpl::list<
      Tags::detail::CosPhi, Tags::detail::CosTheta, Tags::detail::SinPhi,
      Tags::detail::SinTheta, Tags::detail::CartesianCoordinates,
      Tags::detail::CartesianToSphericalJacobian,
      Tags::detail::InverseCartesianToSphericalJacobian,
      ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::SpacetimeMetric<DataVector, 3>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>,
      gh::Tags::Phi<DataVector, 3>, Tags::detail::WorldtubeNormal,
      ::Tags::dt<Tags::detail::WorldtubeNormal>, Tags::detail::NullL,
      ::Tags::dt<Tags::detail::NullL>,
      // for the detail function called at the end
      gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>,
      Tags::detail::AngularDNullL,
      Tags::detail::DLambda<
          gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      Tags::detail::DLambda<
          gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      ::Tags::spacetime_deriv<Tags::detail::RealBondiR, tmpl::size_t<3>,
                              Frame::RadialNull>,
      Tags::detail::DLambda<Tags::detail::DLambda<Tags::detail::RealBondiR>>,
      ::Tags::deriv<Tags::detail::DLambda<Tags::detail::RealBondiR>,
                    tmpl::size_t<2>, Frame::RadialNull>>>
      computation_variables{size};

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>>
      derivative_buffers{size};

  auto& cos_phi = get<Tags::detail::CosPhi>(computation_variables);
  auto& cos_theta = get<Tags::detail::CosTheta>(computation_variables);
  auto& sin_phi = get<Tags::detail::SinPhi>(computation_variables);
  auto& sin_theta = get<Tags::detail::SinTheta>(computation_variables);
  trigonometric_functions_on_swsh_collocation(
      make_not_null(&cos_phi), make_not_null(&cos_theta),
      make_not_null(&sin_phi), make_not_null(&sin_theta), l_max);

  // NOTE: to handle the singular values of polar coordinates, the phi
  // components of all tensors are scaled according to their sin(theta)
  // prefactors.
  // so, any down-index component get<2>(A) represents 1/sin(theta) A_\phi,
  // and any up-index component get<2>(A) represents sin(theta) A^\phi.
  // This holds for Jacobians, and so direct application of the Jacobians
  // brings the factors through.
  auto& unused_cartesian_coords =
      get<Tags::detail::CartesianCoordinates>(computation_variables);
  auto& inverse_cartesian_to_spherical_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          computation_variables);
  auto& cartesian_to_spherical_jacobian =
      get<Tags::detail::CartesianToSphericalJacobian>(computation_variables);
  cartesian_to_spherical_coordinates_and_jacobians(
      make_not_null(&unused_cartesian_coords),
      make_not_null(&cartesian_to_spherical_jacobian),
      make_not_null(&inverse_cartesian_to_spherical_jacobian), cos_phi,
      cos_theta, sin_phi, sin_theta, extraction_radius);

  auto& phi = get<gh::Tags::Phi<DataVector, 3>>(computation_variables);
  auto& dt_spacetime_metric =
      get<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
          computation_variables);
  auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(computation_variables);

  auto& interpolation_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          derivative_buffers);
  auto& eth_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 1>>>(
          derivative_buffers);

  auto& d_cartesian_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);
  auto& d_cartesian_shift =
      get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);
  auto& d_cartesian_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);

  deriv_cartesian_metric_lapse_shift_from_nodes(
      make_not_null(&d_cartesian_spatial_metric),
      make_not_null(&d_cartesian_shift), make_not_null(&d_cartesian_lapse),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      cartesian_spatial_metric, cartesian_dr_spatial_metric, cartesian_shift,
      cartesian_dr_shift, cartesian_lapse, cartesian_dr_lapse,
      inverse_cartesian_to_spherical_jacobian, l_max);

  gh::phi(make_not_null(&phi), cartesian_lapse, d_cartesian_lapse,
          cartesian_shift, d_cartesian_shift, cartesian_spatial_metric,
          d_cartesian_spatial_metric);
  gr::time_derivative_of_spacetime_metric(
      make_not_null(&dt_spacetime_metric), cartesian_lapse, cartesian_dt_lapse,
      cartesian_shift, cartesian_dt_shift, cartesian_spatial_metric,
      cartesian_dt_spatial_metric);
  gr::spacetime_metric(make_not_null(&spacetime_metric), cartesian_lapse,
                       cartesian_shift, cartesian_spatial_metric);
  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(computation_variables);
  inverse_spatial_metric =
      determinant_and_inverse(cartesian_spatial_metric).second;

  auto& dt_worldtube_normal =
      get<::Tags::dt<Tags::detail::WorldtubeNormal>>(computation_variables);
  auto& worldtube_normal =
      get<Tags::detail::WorldtubeNormal>(computation_variables);
  worldtube_normal_and_derivatives(
      make_not_null(&worldtube_normal), make_not_null(&dt_worldtube_normal),
      cos_phi, cos_theta, spacetime_metric, dt_spacetime_metric, sin_phi,
      sin_theta, inverse_spatial_metric);

  auto& du_null_l = get<::Tags::dt<Tags::detail::NullL>>(computation_variables);
  auto& null_l = get<Tags::detail::NullL>(computation_variables);
  null_vector_l_and_derivatives(
      make_not_null(&du_null_l), make_not_null(&null_l), dt_worldtube_normal,
      cartesian_dt_lapse, dt_spacetime_metric, cartesian_dt_shift,
      cartesian_lapse, spacetime_metric, cartesian_shift, worldtube_normal);

  // pass to the next step that is common between the 'modal' input and 'GH'
  // input strategies
  create_bondi_boundary_data(
      bondi_boundary_data, make_not_null(&computation_variables),
      make_not_null(&derivative_buffers), dt_spacetime_metric, phi,
      spacetime_metric, null_l, du_null_l, cartesian_to_spherical_jacobian,
      l_max, extraction_radius);
}

void create_klein_gordon_boundary_data(
    const gsl::not_null<Variables<Tags::characteristic_worldtube_boundary_tags<
        Tags::BoundaryValue, true>>*>
        bondi_boundary_data,
    const tnsr::i<DataVector, 3>& csw_phi, const Scalar<DataVector>& csw_pi,
    const Scalar<DataVector>& csw_psi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3>& shift) {
  klein_gordon_psi_worldtube_data(
      make_not_null(&get<Tags::BoundaryValue<Tags::KleinGordonPsi>>(
          *bondi_boundary_data)),
      csw_psi);

  klein_gordon_pi_worldtube_data(
      make_not_null(
          &get<Tags::BoundaryValue<Tags::KleinGordonPi>>(*bondi_boundary_data)),
      csw_pi, csw_phi, lapse, shift);
}

void create_bondi_boundary_data_from_unnormalized_spec_modes(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        bondi_boundary_data,
    const tnsr::ii<ComplexModalVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dt_spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::I<ComplexModalVector, 3>& shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dt_shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dr_shift_coefficients,
    const Scalar<ComplexModalVector>& lapse_coefficients,
    const Scalar<ComplexModalVector>& dt_lapse_coefficients,
    const Scalar<ComplexModalVector>& dr_lapse_coefficients,
    const double extraction_radius, const size_t l_max) {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // Most allocations required for the full boundary computation are merged into
  // a single, large Variables allocation. There remain a handful of cases in
  // the computational functions called where an intermediate quantity that is
  // not re-used is allocated rather than taking a buffer. These cases are
  // marked with code comments 'Allocation'; In future, allocations are
  // identified as a point to optimize, those buffers may be allocated here and
  // passed as function arguments
  Variables<tmpl::list<
      Tags::detail::CosPhi, Tags::detail::CosTheta, Tags::detail::SinPhi,
      Tags::detail::SinTheta, Tags::detail::CartesianCoordinates,
      Tags::detail::CartesianToSphericalJacobian,
      Tags::detail::InverseCartesianToSphericalJacobian,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>,
      gr::Tags::Shift<DataVector, 3>,
      ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::Shift<DataVector, 3>>, gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      gr::Tags::SpacetimeMetric<DataVector, 3>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>,
      gh::Tags::Phi<DataVector, 3>, Tags::detail::WorldtubeNormal,
      ::Tags::dt<Tags::detail::WorldtubeNormal>, Tags::detail::NullL,
      ::Tags::dt<Tags::detail::NullL>,
      // for the detail function called at the end
      gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>,
      Tags::detail::AngularDNullL,
      Tags::detail::DLambda<
          gr::Tags::SpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      Tags::detail::DLambda<
          gr::Tags::InverseSpacetimeMetric<DataVector, 3, Frame::RadialNull>>,
      ::Tags::spacetime_deriv<Tags::detail::RealBondiR, tmpl::size_t<3>,
                              Frame::RadialNull>,
      Tags::detail::DLambda<Tags::detail::DLambda<Tags::detail::RealBondiR>>,
      ::Tags::deriv<Tags::detail::DLambda<Tags::detail::RealBondiR>,
                    tmpl::size_t<2>, Frame::RadialNull>,
      ::Tags::TempScalar<0, DataVector>>>
      computation_variables{size};

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>>
      derivative_buffers{size};
  auto& cos_phi = get<Tags::detail::CosPhi>(computation_variables);
  auto& cos_theta = get<Tags::detail::CosTheta>(computation_variables);
  auto& sin_phi = get<Tags::detail::SinPhi>(computation_variables);
  auto& sin_theta = get<Tags::detail::SinTheta>(computation_variables);
  trigonometric_functions_on_swsh_collocation(
      make_not_null(&cos_phi), make_not_null(&cos_theta),
      make_not_null(&sin_phi), make_not_null(&sin_theta), l_max);

  // NOTE: to handle the singular values of polar coordinates, the phi
  // components of all tensors are scaled according to their sin(theta)
  // prefactors.
  // so, any down-index component get<2>(A) represents 1/sin(theta) A_\phi,
  // and any up-index component get<2>(A) represents sin(theta) A^\phi.
  // This holds for Jacobians, and so direct application of the Jacobians
  // brings the factors through.
  auto& cartesian_coords =
      get<Tags::detail::CartesianCoordinates>(computation_variables);
  auto& cartesian_to_spherical_jacobian =
      get<Tags::detail::CartesianToSphericalJacobian>(computation_variables);
  auto& inverse_cartesian_to_spherical_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          computation_variables);
  cartesian_to_spherical_coordinates_and_jacobians(
      make_not_null(&cartesian_coords),
      make_not_null(&cartesian_to_spherical_jacobian),
      make_not_null(&inverse_cartesian_to_spherical_jacobian), cos_phi,
      cos_theta, sin_phi, sin_theta, extraction_radius);

  auto& cartesian_spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(computation_variables);
  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(computation_variables);
  auto& d_cartesian_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);
  auto& dt_cartesian_spatial_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(
          computation_variables);
  auto& interpolation_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          derivative_buffers);
  Scalar<SpinWeighted<ComplexModalVector, 0>> interpolation_modal_buffer{size};
  auto& eth_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 1>>>(
          derivative_buffers);
  auto& radial_correction_factor =
      get<::Tags::TempScalar<0, DataVector>>(computation_variables);
  cartesian_spatial_metric_and_derivatives_from_unnormalized_spec_modes(
      make_not_null(&cartesian_spatial_metric),
      make_not_null(&inverse_spatial_metric),
      make_not_null(&d_cartesian_spatial_metric),
      make_not_null(&dt_cartesian_spatial_metric),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      make_not_null(&radial_correction_factor), spatial_metric_coefficients,
      dr_spatial_metric_coefficients, dt_spatial_metric_coefficients,
      inverse_cartesian_to_spherical_jacobian, cartesian_coords, l_max);

  auto& cartesian_shift =
      get<gr::Tags::Shift<DataVector, 3>>(computation_variables);
  auto& d_cartesian_shift =
      get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);
  auto& dt_cartesian_shift =
      get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(computation_variables);

  cartesian_shift_and_derivatives_from_unnormalized_spec_modes(
      make_not_null(&cartesian_shift), make_not_null(&d_cartesian_shift),
      make_not_null(&dt_cartesian_shift),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      shift_coefficients, dr_shift_coefficients, dt_shift_coefficients,
      inverse_cartesian_to_spherical_jacobian, radial_correction_factor, l_max);

  auto& cartesian_lapse =
      get<gr::Tags::Lapse<DataVector>>(computation_variables);
  auto& d_cartesian_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);
  auto& dt_cartesian_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(computation_variables);
  cartesian_lapse_and_derivatives_from_unnormalized_spec_modes(
      make_not_null(&cartesian_lapse), make_not_null(&d_cartesian_lapse),
      make_not_null(&dt_cartesian_lapse),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      lapse_coefficients, dr_lapse_coefficients, dt_lapse_coefficients,
      inverse_cartesian_to_spherical_jacobian, radial_correction_factor, l_max);

  auto& phi = get<gh::Tags::Phi<DataVector, 3>>(computation_variables);
  auto& dt_spacetime_metric =
      get<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
          computation_variables);
  auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(computation_variables);
  gh::phi(make_not_null(&phi), cartesian_lapse, d_cartesian_lapse,
          cartesian_shift, d_cartesian_shift, cartesian_spatial_metric,
          d_cartesian_spatial_metric);
  gr::time_derivative_of_spacetime_metric(
      make_not_null(&dt_spacetime_metric), cartesian_lapse, dt_cartesian_lapse,
      cartesian_shift, dt_cartesian_shift, cartesian_spatial_metric,
      dt_cartesian_spatial_metric);
  gr::spacetime_metric(make_not_null(&spacetime_metric), cartesian_lapse,
                       cartesian_shift, cartesian_spatial_metric);

  auto& dt_worldtube_normal =
      get<::Tags::dt<Tags::detail::WorldtubeNormal>>(computation_variables);
  auto& worldtube_normal =
      get<Tags::detail::WorldtubeNormal>(computation_variables);
  worldtube_normal_and_derivatives(
      make_not_null(&worldtube_normal), make_not_null(&dt_worldtube_normal),
      cos_phi, cos_theta, spacetime_metric, dt_spacetime_metric, sin_phi,
      sin_theta, inverse_spatial_metric);

  auto& du_null_l = get<::Tags::dt<Tags::detail::NullL>>(computation_variables);
  auto& null_l = get<Tags::detail::NullL>(computation_variables);
  null_vector_l_and_derivatives(
      make_not_null(&du_null_l), make_not_null(&null_l), dt_worldtube_normal,
      dt_cartesian_lapse, dt_spacetime_metric, dt_cartesian_shift,
      cartesian_lapse, spacetime_metric, cartesian_shift, worldtube_normal);

  // pass to the next step that is common between the 'modal' input and 'GH'
  // input strategies
  create_bondi_boundary_data(
      bondi_boundary_data, make_not_null(&computation_variables),
      make_not_null(&derivative_buffers), dt_spacetime_metric, phi,
      spacetime_metric, null_l, du_null_l, cartesian_to_spherical_jacobian,
      l_max, extraction_radius);
}
}  // namespace Cce
