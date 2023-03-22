// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

/// @{
/*!
 * \brief Converts real spherical harmonic coefficients of degree l into a
 * symmetric trace-free tensor of rank l.
 *
 * \details Spherical harmonics of degree l are equivalent to symmetric
 * trace-free tensors of rank l. This equivalence and the transformation is
 * given e.g. in \cite Thorne1980, Eqs. (2.10) - (2.14). The conversion
 * coefficients are hard-coded to numerical precision and implemented up to
 * order l=2.
 *
 * The spherical harmonic coefficients are expected to be sorted with ascending
 * m, i.e. (-m, -m+1, ... , m)
 */
Scalar<double> ylm_to_stf_0(const ModalVector& l0_coefs);

template <typename Frame>
tnsr::i<double, 3, Frame> ylm_to_stf_1(const ModalVector& l1_coefs);

template <typename Frame>
tnsr::ii<double, 3, Frame> ylm_to_stf_2(const ModalVector& l2_coefs);
/// @}
