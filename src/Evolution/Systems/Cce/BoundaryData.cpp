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
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
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
    const gsl::not_null<jacobian_tensor*> cartesian_to_spherical_jacobian,
    const gsl::not_null<inverse_jacobian_tensor*>
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

void null_metric_and_derivative(
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::RadialNull>*>
        du_null_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::RadialNull>*>
        null_metric,
    const jacobian_tensor& cartesian_to_spherical_jacobian,
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
    const tnsr::iA<DataVector, 3>& angular_d_null_l,
    const jacobian_tensor& cartesian_to_spherical_jacobian,
    const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::A<DataVector, 3>& du_null_l,
    const tnsr::AA<DataVector, 3>& inverse_null_metric,
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
        angular_d_null_l.get(A + 1, 1) * get<1, 0>(spacetime_metric) +
        angular_d_null_l.get(A + 1, 0) * get<0, 0>(spacetime_metric);
    for (size_t k = 1; k < 3; ++k) {
      dlambda_null_metric->get(0, A + 2) +=
          cartesian_to_spherical_jacobian.get(A + 1, k) *
              (get<0>(du_null_l) * spacetime_metric.get(k + 1, 0) +
               get<0>(null_l) * dt_spacetime_metric.get(k + 1, 0)) +
          angular_d_null_l.get(A + 1, k + 1) * spacetime_metric.get(k + 1, 0);
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
              (angular_d_null_l.get(A + 1, a) *
                   cartesian_to_spherical_jacobian.get(B + 1, i) +
               angular_d_null_l.get(B + 1, a) *
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
}  // namespace Cce
