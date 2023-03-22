// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/YlmToStf.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/TMPL.hpp"

Scalar<double> ylm_to_stf_0(const ModalVector& l0_coefs) {
  ASSERT(l0_coefs.size() == 1,
         "Expected 1 spherical harmonic coefficient for l=0");
  return Scalar<double>{l0_coefs[0] * M_2_SQRTPI * 0.25};
}

template <typename Frame>
tnsr::i<double, 3, Frame> ylm_to_stf_1(const ModalVector& l1_coefs) {
  ASSERT(l1_coefs.size() == 3,
         "Expected 3 spherical harmonic coefficients for l=1");
  // sqrt(3 / pi) / 2
  const double coef = 0.4886025119029199;
  tnsr::i<double, 3, Frame> result{};
  get<0>(result) = coef * l1_coefs[2];
  get<1>(result) = coef * l1_coefs[0];
  get<2>(result) = coef * l1_coefs[1];
  return result;
}

template <typename Frame>
tnsr::ii<double, 3, Frame> ylm_to_stf_2(const ModalVector& l2_coefs) {
  ASSERT(l2_coefs.size() == 5,
         "Expected 5 spherical harmonic coefficients for l=2");
  // sqrt(5) / (4 * sqrt(pi))
  const double coef1 = 0.3153915652525199;
  // sqrt(15) / (4 * sqrt(pi))
  const double coef2 = 0.5462742152960396;

  tnsr::ii<double, 3, Frame> result{};
  get<0, 0>(result) = -coef1 * l2_coefs[2] + coef2 * l2_coefs[4];
  get<1, 0>(result) = coef2 * l2_coefs[0];
  get<1, 1>(result) = -coef1 * l2_coefs[2] - coef2 * l2_coefs[4];
  get<2, 0>(result) = coef2 * l2_coefs[3];
  get<2, 1>(result) = coef2 * l2_coefs[1];
  get<2, 2>(result) = 2. * coef1 * l2_coefs[2];
  return result;
}

template tnsr::i<double, 3, Frame::Grid> ylm_to_stf_1<Frame::Grid>(
    const ModalVector& l1_coefs);
template tnsr::i<double, 3, Frame::Inertial> ylm_to_stf_1<Frame::Inertial>(
    const ModalVector& l1_coefs);
template tnsr::ii<double, 3, Frame::Grid> ylm_to_stf_2<Frame::Grid>(
    const ModalVector& l2_coefs);
template tnsr::ii<double, 3, Frame::Inertial> ylm_to_stf_2<Frame::Inertial>(
    const ModalVector& l2_coefs);
