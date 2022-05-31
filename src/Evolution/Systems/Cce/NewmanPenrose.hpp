// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {

/// \cond
namespace Tags {
  struct LMax;
} // namespace Tags
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

/*!
 * \brief Transform `Tags::BondiJ` from the partially flat coordinates
 * to the Cauchy coordinates.
 *
 * \details The spin-2 quantity \f$\hat J\f$ transforms as
 * \f{align*}{
 * J = \frac{1}{4 \omega^2} (\bar d^2 \hat J + c^2 \bar{\hat J}
 * + 2 c \bar d \hat K )
 * \f}
 *
 * with
 * \f{align*}{
 * \hat K = \sqrt{1+\hat J \bar{\hat J}}
 * \f}
 */
struct TransformBondiJToCauchyCoords {
  using return_tags = tmpl::list<Tags::BondiJCauchyView>;
  using argument_tags = tmpl::list<
      Tags::CauchyGaugeC, Tags::BondiJ, Tags::CauchyGaugeD,
      Tags::CauchyGaugeOmega,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::PartiallyFlatAngularCoords>,
      Tags::LMax>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          cauchy_view_volume_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_cauchy_c,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& volume_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_cauchy_d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cauchy,
      const Spectral::Swsh::SwshInterpolator& interpolator,
      const size_t l_max);
};

/*!
 * \brief Compute the Weyl scalar \f$\Psi_0\f$ in the volume for the purpose
 * of CCM, the quantity is in the Cauchy coordinates.
 *
 * \details The Weyl scalar \f$\Psi_0\f$ is given by:
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
 *
 * The quantities above are all in the Cauchy coordinates, where \f$K\f$ is
 * updated from \f$J\f$ and \f$\bar J\f$, \f$(1-y)\f$ is invariant under
 * the coordinate transformation. \f$r\f$ transforms as
 *
 * \f{align*}{
 * r = \omega \hat r
 * \f}
 */
template <>
struct VolumeWeyl<Tags::Psi0Match> {
  using return_tags = tmpl::list<Tags::Psi0Match>;
  using argument_tags =
      tmpl::list<Tags::BondiJCauchyView, Tags::Dy<Tags::BondiJCauchyView>,
                 Tags::Dy<Tags::Dy<Tags::BondiJCauchyView>>,
                 Tags::BoundaryValue<Tags::BondiR>, Tags::OneMinusY,
                 Tags::LMax>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi_0,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j_cauchy,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j_cauchy,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_dy_j_cauchy,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r_cauchy,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const size_t l_max);
};

/*!
 * \brief Compute the Weyl scalar \f$\Psi_0\f$ and its radial derivative
 * \f$\partial_\lambda \Psi_0\f$ on the inner boundary of CCE domain.
 * The quantities are in the Cauchy coordinates.
 *
 * \details The radial derivative of the Weyl scalar \f$\partial_\lambda
 * \Psi_0\f$ is given by
 *
 * \f{align*}{
 * \partial_\lambda \Psi_0 = \frac{(1-y)^2}{2r}e^{-2\beta}
 * \partial_y \Psi_0
 * \f}
 *
 * Note that \f$(1-y)\f$, \f$r\f$, and \f$\beta\f$ are in the Cauchy
 * coordinates, where \f$(1-y)\f$ is invariant under the coordinate
 * transformation, while \f$r\f$ and \f$\beta\f$ transform as
 *
 * \f{align*}{
 * &r = \omega \hat r
 * & \beta = \hat \beta - \frac{1}{2} \log \omega
 * \f}
 */
struct InnerBoundaryWeyl {
  using return_tags =
      tmpl::list<Tags::BoundaryValue<Tags::Psi0Match>,
                 Tags::BoundaryValue<Tags::Dlambda<Tags::Psi0Match>>>;
  using argument_tags =
      tmpl::list<Tags::Psi0Match, Tags::Dy<Tags::Psi0Match>, Tags::OneMinusY,
                 Tags::BoundaryValue<Tags::BondiR>,
                 Tags::BoundaryValue<Tags::BondiBeta>, Tags::LMax>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi_0_boundary,
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          dlambda_psi_0_boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& psi_0,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_psi_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r_cauchy,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_beta_cauchy,
      const size_t l_max);
};
}  // namespace Cce
