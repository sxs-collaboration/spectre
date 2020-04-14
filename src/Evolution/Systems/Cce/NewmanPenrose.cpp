// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/NewmanPenrose.hpp"

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

void weyl_psi0_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> psi_0,
    const SpinWeighted<ComplexDataVector, 2>& bondi_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_dy_j,
    const SpinWeighted<ComplexDataVector, 0>& bondi_k,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) noexcept {
  // note: intential use of expression template instead of allocation
  const auto dy_beta = 0.125 * one_minus_y *
                 (dy_j * conj(dy_j) -
                  0.25 * square(bondi_j * conj(dy_j) + conj(bondi_j) * dy_j) /
                      square(bondi_k));
  *psi_0 = pow<4>(one_minus_y) * 0.0625 / (square(bondi_r) * bondi_k) *
           (dy_beta * ((1.0 + bondi_k) * dy_j -
                       square(bondi_j) * conj(dy_j) / (1.0 + bondi_k)) -
            0.5 * (1.0 + bondi_k) * dy_dy_j +
            0.5 * square(bondi_j) * conj(dy_dy_j) / (1.0 + bondi_k) +
            (-0.25 * bondi_j *
                 (square(conj(bondi_j)) * square(dy_j) +
                  square(bondi_j) * square(conj(dy_j))) +
             0.5 * bondi_j * (1.0 + square(bondi_k)) * dy_j * conj(dy_j)) /
                square(bondi_k));
}

void VolumeWeyl<Tags::Psi0>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi_0,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) noexcept {
  weyl_psi0_impl(make_not_null(&get(*psi_0)), get(bondi_j), get(dy_j),
                 get(dy_dy_j), get(bondi_k), get(bondi_r), get(one_minus_y));
}

}  // namespace Cce
