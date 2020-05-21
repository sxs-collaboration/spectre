// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Evolution/Systems/Cce/AnalyticSolutions/AnalyticDataHelpers.hpp"

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce::Solutions::TestHelpers {

tuples::TaggedTuple<Tags::BondiBeta, Tags::BondiU, Tags::BondiW, Tags::BondiJ>
extract_bondi_scalars_from_cartesian_metric(
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const CartesianiSphericalJ& inverse_jacobian,
    const double extraction_radius) noexcept {
  const auto inverse_cartesian_metric =
      determinant_and_inverse(spacetime_metric).second;
  tnsr::AA<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>
      inverse_spherical_metric{get<0, 0>(spacetime_metric).size(), 0.0};
  get<0, 0>(inverse_spherical_metric) = get<0, 0>(inverse_cartesian_metric);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      inverse_spherical_metric.get(0, i + 1) +=
          inverse_jacobian.get(k, i) * inverse_cartesian_metric.get(0, k + 1);
    }
    for (size_t j = i; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          inverse_spherical_metric.get(i + 1, j + 1) +=
              inverse_jacobian.get(k, i) * inverse_jacobian.get(l, j) *
              inverse_cartesian_metric.get(k + 1, l + 1);
        }
      }
    }
  }
  get<0, 0>(inverse_spherical_metric) +=
      -2.0 * get<0, 1>(inverse_spherical_metric) +
      get<1, 1>(inverse_spherical_metric);
  for(size_t i = 0; i < 3; ++i) {
    inverse_spherical_metric.get(0, i) -= inverse_spherical_metric.get(1, i);
  }
  Scalar<SpinWeighted<ComplexDataVector, 0>> bondi_beta{
      get<0, 0>(spacetime_metric).size()};
  get(bondi_beta).data() = std::complex<double>(1.0, 0.0) * 0.5 *
                           log(-get<0, 1>(inverse_spherical_metric));
  Scalar<SpinWeighted<ComplexDataVector, 0>> bondi_w{
      get<0, 0>(spacetime_metric).size()};
  get(bondi_w) = std::complex<double>(1.0, 0.0) *
                 (-get<1, 1>(inverse_spherical_metric) /
                      get<0, 1>(inverse_spherical_metric) -
                  1.0) /
                 extraction_radius;
  Scalar<SpinWeighted<ComplexDataVector, 1>> bondi_u{
      get<0, 0>(spacetime_metric).size()};
  // note that the jacobian components are provided in 'pfaffian' form - so the
  // extra factors of sin(theta) are omitted.
  get(bondi_u) =
      -(std::complex<double>(1.0, 0.0) * get<1, 2>(inverse_spherical_metric) +
        std::complex<double>(0.0, 1.0) * get<1, 3>(inverse_spherical_metric)) /
      get<0, 1>(inverse_spherical_metric);
  Scalar<SpinWeighted<ComplexDataVector, 2>> bondi_j{
      get<0, 0>(spacetime_metric).size()};
  get(bondi_j) =
      -0.5 * square(extraction_radius) *
      (get<2, 2>(inverse_spherical_metric) -
       get<3, 3>(inverse_spherical_metric) +
       std::complex<double>(0.0, 2.0) * get<2, 3>(inverse_spherical_metric));
  return {bondi_beta, bondi_u, bondi_w, bondi_j};
}

tuples::TaggedTuple<::Tags::dt<Tags::BondiBeta>, ::Tags::dt<Tags::BondiU>,
                    ::Tags::dt<Tags::BondiW>, ::Tags::dt<Tags::BondiJ>>
extract_dt_bondi_scalars_from_cartesian_metric(
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const CartesianiSphericalJ& inverse_jacobian,
    const double extraction_radius) noexcept {
  const auto inverse_cartesian_metric =
      determinant_and_inverse(spacetime_metric).second;
  tnsr::AA<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>
      dt_inverse_cartesian_metric{get<0, 0>(spacetime_metric).size(), 0.0};
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      for (size_t c = 0; c < 4; ++c) {
        for (size_t d = 0; d < 4; ++d) {
          dt_inverse_cartesian_metric.get(a, b) +=
              -inverse_cartesian_metric.get(a, c) *
              inverse_cartesian_metric.get(b, d) *
              dt_spacetime_metric.get(c, d);
        }
      }
    }
  }

  tnsr::AA<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>
      inverse_spherical_metric{get<0, 0>(spacetime_metric).size(), 0.0};
  tnsr::AA<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>
      dt_inverse_spherical_metric{get<0, 0>(spacetime_metric).size(), 0.0};
  get<0, 0>(inverse_spherical_metric) = get<0, 0>(inverse_cartesian_metric);
  get<0, 0>(dt_inverse_spherical_metric) =
      get<0, 0>(dt_inverse_cartesian_metric);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      inverse_spherical_metric.get(0, i + 1) +=
          inverse_jacobian.get(k, i) * inverse_cartesian_metric.get(0, k + 1);
      dt_inverse_spherical_metric.get(0, i + 1) +=
          inverse_jacobian.get(k, i) *
          dt_inverse_cartesian_metric.get(0, k + 1);
    }
    for (size_t j = i; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          inverse_spherical_metric.get(i + 1, j + 1) +=
              inverse_jacobian.get(k, i) * inverse_jacobian.get(l, j) *
              inverse_cartesian_metric.get(k + 1, l + 1);
          dt_inverse_spherical_metric.get(i + 1, j + 1) +=
              inverse_jacobian.get(k, i) * inverse_jacobian.get(l, j) *
              dt_inverse_cartesian_metric.get(k + 1, l + 1);
        }
      }
    }
  }
  get<0, 0>(inverse_spherical_metric) +=
      -2.0 * get<0, 1>(inverse_spherical_metric) +
      get<1, 1>(inverse_spherical_metric);
  get<0, 0>(dt_inverse_spherical_metric) +=
      -2.0 * get<0, 1>(dt_inverse_spherical_metric) +
      get<1, 1>(dt_inverse_spherical_metric);
  for(size_t i = 0; i < 3; ++i) {
    inverse_spherical_metric.get(0, i) -= inverse_spherical_metric.get(1, i);
    dt_inverse_spherical_metric.get(0, i) -=
        dt_inverse_spherical_metric.get(1, i);
  }

  // The formulas for these scalars can be determined by differentiating the
  // forms in `extract_bondi_scalars_from_cartesian_metric`
  Scalar<SpinWeighted<ComplexDataVector, 0>> dt_bondi_beta{
      get<0, 0>(spacetime_metric).size()};
  get(dt_bondi_beta) = -std::complex<double>(0.5, 0.0) *
                       get<0, 1>(dt_inverse_spherical_metric) /
                       get<0, 1>(inverse_spherical_metric);
  Scalar<SpinWeighted<ComplexDataVector, 0>> dt_bondi_w{
      get<0, 0>(spacetime_metric).size()};
  get(dt_bondi_w) = std::complex<double>(1.0, 0.0) *
                    (-get<1, 1>(dt_inverse_spherical_metric) /
                         get<0, 1>(inverse_spherical_metric) +
                     get<1, 1>(inverse_spherical_metric) *
                         get<0, 1>(dt_inverse_spherical_metric) /
                         square(get<0, 1>(inverse_spherical_metric))) /
                    extraction_radius;
  Scalar<SpinWeighted<ComplexDataVector, 1>> dt_bondi_u{
      get<0, 0>(spacetime_metric).size()};
  // note that the jacobian components are provided in 'pfaffian' form - so the
  // extra factors of sin(theta) are omitted.
  get(dt_bondi_u) =
      -(std::complex<double>(1.0, 0.0) *
            get<1, 2>(dt_inverse_spherical_metric) +
        std::complex<double>(0.0, 1.0) *
            get<1, 3>(dt_inverse_spherical_metric)) /
          get<0, 1>(inverse_spherical_metric) +
      (std::complex<double>(1.0, 0.0) * get<1, 2>(inverse_spherical_metric) +
       std::complex<double>(0.0, 1.0) * get<1, 3>(inverse_spherical_metric)) *
          get<0, 1>(dt_inverse_spherical_metric) /
          square(get<0, 1>(inverse_spherical_metric));
  Scalar<SpinWeighted<ComplexDataVector, 2>> dt_bondi_j{
      get<0, 0>(spacetime_metric).size()};
  get(dt_bondi_j) = -square(extraction_radius) *
                    std::complex<double>(0.5, 0.0) *
                    (get<2, 2>(dt_inverse_spherical_metric) -
                     get<3, 3>(dt_inverse_spherical_metric) +
                     std::complex<double>(0.0, 2.0) *
                         (get<2, 3>(dt_inverse_spherical_metric)));
  return {dt_bondi_beta, dt_bondi_u, dt_bondi_w, dt_bondi_j};
}

tuples::TaggedTuple<Tags::Dr<Tags::BondiBeta>, Tags::Dr<Tags::BondiU>,
                    Tags::Dr<Tags::BondiW>, Tags::Dr<Tags::BondiJ>>
extract_dr_bondi_scalars_from_cartesian_metric(
    const tnsr::aa<DataVector, 3>& dr_spacetime_metric,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const CartesianiSphericalJ& inverse_jacobian,
    const CartesianiSphericalJ& dr_inverse_jacobian,
    const double extraction_radius) noexcept {
  const auto inverse_cartesian_metric =
      determinant_and_inverse(spacetime_metric).second;
  tnsr::AA<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>
      dr_inverse_cartesian_metric{get<0, 0>(spacetime_metric).size(), 0.0};
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      for (size_t c = 0; c < 4; ++c) {
        for (size_t d = 0; d < 4; ++d) {
          dr_inverse_cartesian_metric.get(a, b) +=
              -inverse_cartesian_metric.get(a, c) *
              inverse_cartesian_metric.get(b, d) *
              dr_spacetime_metric.get(c, d);
        }
      }
    }
  }

  tnsr::AA<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>
      inverse_spherical_metric{get<0, 0>(spacetime_metric).size(), 0.0};
  tnsr::AA<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>
      dr_inverse_spherical_metric{get<0, 0>(spacetime_metric).size(), 0.0};
  get<0, 0>(inverse_spherical_metric) = get<0, 0>(inverse_cartesian_metric);
  get<0, 0>(dr_inverse_spherical_metric) =
      get<0, 0>(dr_inverse_cartesian_metric);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      inverse_spherical_metric.get(0, i + 1) +=
          inverse_jacobian.get(k, i) * inverse_cartesian_metric.get(0, k + 1);
      dr_inverse_spherical_metric.get(0, i + 1) +=
          inverse_jacobian.get(k, i) *
              dr_inverse_cartesian_metric.get(0, k + 1) +
          dr_inverse_jacobian.get(k, i) *
              inverse_cartesian_metric.get(0, k + 1);
    }
    for (size_t j = i; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          inverse_spherical_metric.get(i + 1, j + 1) +=
              inverse_jacobian.get(k, i) * inverse_jacobian.get(l, j) *
              inverse_cartesian_metric.get(k + 1, l + 1);
          dr_inverse_spherical_metric.get(i + 1, j + 1) +=
              inverse_jacobian.get(k, i) * inverse_jacobian.get(l, j) *
                  dr_inverse_cartesian_metric.get(k + 1, l + 1) +
              dr_inverse_jacobian.get(k, i) * inverse_jacobian.get(l, j) *
                  inverse_cartesian_metric.get(k + 1, l + 1) +
              inverse_jacobian.get(k, i) * dr_inverse_jacobian.get(l, j) *
                  inverse_cartesian_metric.get(k + 1, l + 1);
        }
      }
    }
  }

  get<0, 0>(inverse_spherical_metric) +=
      -2.0 * get<0, 1>(inverse_spherical_metric) +
      get<1, 1>(inverse_spherical_metric);
  get<0, 0>(dr_inverse_spherical_metric) +=
      -2.0 * get<0, 1>(dr_inverse_spherical_metric) +
      get<1, 1>(dr_inverse_spherical_metric);
  for (size_t i = 0; i < 3; ++i) {
    inverse_spherical_metric.get(0, i) -= inverse_spherical_metric.get(1, i);
    dr_inverse_spherical_metric.get(0, i) -=
        dr_inverse_spherical_metric.get(1, i);
  }

  // The formulas for these scalars can be determined by differentiating the
  // forms in `extract_bondi_scalars_from_cartesian_metric`
  Scalar<SpinWeighted<ComplexDataVector, 0>> dr_bondi_beta{
      get<0, 0>(spacetime_metric).size()};
  get(dr_bondi_beta) = -std::complex<double>(0.5, 0.0) *
                       get<0, 1>(dr_inverse_spherical_metric) /
                       get<0, 1>(inverse_spherical_metric);
  Scalar<SpinWeighted<ComplexDataVector, 0>> dr_bondi_w{
      get<0, 0>(spacetime_metric).size()};
  get(dr_bondi_w) = std::complex<double>(1.0, 0.0) *
                    ((-get<1, 1>(dr_inverse_spherical_metric) /
                          get<0, 1>(inverse_spherical_metric) +
                      get<1, 1>(inverse_spherical_metric) *
                          get<0, 1>(dr_inverse_spherical_metric) /
                          square(get<0, 1>(inverse_spherical_metric))) /
                         extraction_radius +
                     (get<1, 1>(inverse_spherical_metric) /
                          get<0, 1>(inverse_spherical_metric) +
                      1.0) /
                         square(extraction_radius));
  Scalar<SpinWeighted<ComplexDataVector, 1>> dr_bondi_u{
      get<0, 0>(spacetime_metric).size()};
  // note that the jacobian components are provided in 'pfaffian' form - so the
  // extra factors of sin(theta) are omitted.
  get(dr_bondi_u) =
      -(std::complex<double>(1.0, 0.0) *
            get<1, 2>(dr_inverse_spherical_metric) +
        std::complex<double>(0.0, 1.0) *
            get<1, 3>(dr_inverse_spherical_metric)) /
          get<0, 1>(inverse_spherical_metric) +
      (std::complex<double>(1.0, 0.0) * get<1, 2>(inverse_spherical_metric) +
       std::complex<double>(0.0, 1.0) * get<1, 3>(inverse_spherical_metric)) *
          get<0, 1>(dr_inverse_spherical_metric) /
          square(get<0, 1>(inverse_spherical_metric));
  Scalar<SpinWeighted<ComplexDataVector, 2>> dr_bondi_j{
      get<0, 0>(spacetime_metric).size()};
  get(dr_bondi_j) =
      -0.5 * square(extraction_radius) *
          (get<2, 2>(dr_inverse_spherical_metric) -
           get<3, 3>(dr_inverse_spherical_metric) +
           std::complex<double>(0.0, 2.0) *
               (get<2, 3>(dr_inverse_spherical_metric))) +
      -extraction_radius * (get<2, 2>(inverse_spherical_metric) -
                            get<3, 3>(inverse_spherical_metric) +
                            std::complex<double>(0.0, 2.0) *
                                (get<2, 3>(inverse_spherical_metric)));
  return {dr_bondi_beta, dr_bondi_u, dr_bondi_w, dr_bondi_j};
}

}  // namespace Cce::Solutions::TestHelpers
