// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {

/// \cond
template <typename Tag>
struct VolumeWeyl;
/// \endcond

/*!
 * \brief Compute the Weyl scalar \f$\Psi_0\f$ in the volume according to a
 * standard set of Newman-Penrose vectors.
 *
 * \details The Bondi forms of the Newman-Penrose vectors that are needed for
 * \f$\Psi_0\f$ are:
 *
 * \f{align}{
 * \mathbf{l} &= \partial_r / \sqrt{2}\\
 * \mathbf{m} &= \frac{-1}{2 r} \left(\sqrt{1 + K} q^A \partial_A -
 *   \frac{J}{\sqrt{1 + K}}\bar{q}^A \partial_A \right)
 * \f}
 *
 * Then, we may compute \f$\Psi_0 =  l^\alpha m^\beta l^\mu m^\nu C_{\alpha
 * \beta \mu \nu}\f$ from the Bondi system, giving
 *
 * \f{align*}{
 * \Psi_0 = \frac{(1 - y)^4}{16 r^2 K}
 * \bigg[& \partial_y \beta \left((1 + K) (\partial_y J)
 * - \frac{J^2 \partial_y \bar J}{1 + K}\right)
 * - \frac{1}{2} (1 + K) (\partial_y^2 J)
 * + \frac{J^2 \partial_y^2 \bar J}{2(K + 1)}\\
 * & + \frac{1}{K^2} \left(- \frac{1}{4} J \left(\bar{J}^2 \left(\partial_y
 * J\right)^2 + J^2 \left(\partial_y \bar J\right)^2\right)
 * + \frac{1 + K^2}{2} J (\partial_y J) (\partial_y \bar J)
 * \right)\bigg].
 * \f}
 */
template <>
struct VolumeWeyl<Tags::Psi0> {
  using return_tags = tmpl::list<Tags::Psi0>;
  using argument_tags = tmpl::list<Tags::BondiJ, Tags::Dy<Tags::BondiJ>,
                                   Tags::Dy<Tags::Dy<Tags::BondiJ>>,
                                   Tags::BondiK, Tags::BondiR, Tags::OneMinusY>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi_0,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);
};
}  // namespace Cce
