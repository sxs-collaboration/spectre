// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

/// The set of tags that should be calculated before the initial data is
/// computed on the first hypersurface.
using gauge_adjustments_setup_tags =
    tmpl::list<Tags::BondiR, Tags::BondiJ, Tags::Dr<Tags::BondiJ>>;

template <typename Tag>
struct GaugeAdjustedBoundaryValue;

/*!
 * \brief Computes the evolution gauge Bondi \f$\hat R\f$ on the worldtube from
 * Cauchy gauge quantities
 *
 * \details The evolution gauge Bondi \f$\hat R\f$ obeys:
 *
 * \f{align*}{
 * \hat R = \hat \omega R(\hat x^{\hat A}),
 * \f}
 *
 * where the evaluation of \f$R\f$ at \f$\hat x^{\hat A}\f$ requires an
 * interpolation to the evolution coordinates, and \f$\hat \omega\f$ is the
 * conformal factor associated with the angular part of the gauge
 * transformation.
 */
template <>
struct GaugeAdjustedBoundaryValue<Tags::BondiR> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::BondiR>, Tags::GaugeOmega,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& cauchy_gauge_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Spectral::Swsh::SwshInterpolator& interpolator) noexcept;
};

/*!
 * \brief Computes the evolution gauge \f$\partial_{\hat u} \hat R / \hat R\f$
 * on the worldtube.
 *
 * \details The evolution gauge quantity \f$ \partial_{\hat u} \hat R / \hat
 * R\f$ obeys
 *
 * \f{align*}{
 *  \frac{\partial_{\hat u} \hat R}{ \hat R}
 * = \frac{\partial_u R (\hat x^{\hat A})}{R(\hat x^{\hat A})}
 * + \frac{\partial_{\hat u} \hat \omega}{\hat \omega}
 * + \frac{\mathcal U^{(0)} \bar \eth R(\hat x^{\hat A})
 * + \bar{\mathcal U}^{(0)} \eth R(\hat x^{\hat A}) }{2 R(\hat x^{\hat A})}
 * \f}
 *
 * note that the terms proportional to \f$\eth R\f$ or its conjugate arise from
 * the conversion between \f$\partial_u\f$ and \f$\partial_{\hat u}f\f$. The
 * right-hand side quantities with explicit \f$\hat x\f$ require interpolation.
 * \f$\mathcal U^{(0)}\f$ is the asymptotic quantity determined by
 * `GaugeUpdateTimeDerivatives`.
 */
template <>
struct GaugeAdjustedBoundaryValue<Tags::DuRDividedByR> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::DuRDividedByR>, Tags::BondiUAtScri,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>, Tags::GaugeOmega,
      Tags::Du<Tags::GaugeOmega>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::LMax>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_du_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&
          cauchy_gauge_du_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u_at_scri,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega,
      const Spectral::Swsh::SwshInterpolator& interpolator,
      size_t l_max) noexcept;
};

/*!
 * \brief Computes the evolution gauge quantity \f$\hat J\f$ on the worldtube
 *
 * \details The evolution gauge quantity \f$\hat J\f$ obeys
 *
 * \f{align*}{
 * \hat J = \frac{1}{4 \hat{\omega}^2} \left( \bar{\hat d}^2  J(\hat x^{\hat A})
 * + \hat c^2 \bar J(\hat x^{\hat A})
 * + 2 \hat c \bar{\hat d} K(\hat x^{\hat A}) \right)
 * \f}
 *
 * Where \f$\hat c\f$ and \f$\hat d\f$ are the spin-weighted angular Jacobian
 * factors computed by `GaugeUpdateJacobianFromCoords`, and \f$\hat \omega\f$ is
 * the conformal factor associated with the angular coordinate transformation.
 * Note that the right-hand sides with explicit \f$\hat x^{\hat A}\f$ dependence
 * must be interpolated and that \f$K = \sqrt{1 + J \bar J}\f$.
 */
template <>
struct GaugeAdjustedBoundaryValue<Tags::BondiJ> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::BondiJ>, Tags::GaugeC, Tags::GaugeD,
      Tags::GaugeOmega,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          evolution_gauge_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& cauchy_gauge_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Spectral::Swsh::SwshInterpolator& interpolator) noexcept;
};

/*!
 * \brief Computes the evolution gauge quantity \f$\partial_{\hat r} \hat J\f$
 * on the worldtube
 *
 * \details The evolution gauge quantity \f$\partial_{\hat r} \hat J\f$ is
 * determined from \f$\partial_{\hat r} = \frac{\partial_r}{\hat \omega}\f$ and
 * the expression for \f$\hat J\f$ given in the documentation for
 * `GaugeAdjustedBoundaryValue<Tags::BondiJ>`
 */
template <>
struct GaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::BondiJ>>>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::BondiJ>, Tags::GaugeC, Tags::GaugeD,
      Tags::GaugeOmega,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::LMax>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          evolution_gauge_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& cauchy_gauge_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& cauchy_gauge_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Spectral::Swsh::SwshInterpolator& interpolator,
      size_t l_max) noexcept;
};

/*!
 * \brief Computes the evolution gauge quantity \f$\hat \beta\f$ on the
 * worldtube
 *
 * \details The evolution gauge quantity \f$\hat \beta\f$ obeys
 *
 * \f{align*}{
 * e^{2 \hat \beta} = e^{2 \beta(\hat x^{\hat A})} / \hat \omega.
 * \f}
 *
 * The explicit evaluation at \f$\hat x^{\hat A}\f$ on the right-hand side
 * indicates the requirement of an interpolation step, and \f$\hat \omega\f$ is
 * the conformal factor associated with the angular transformation.
 */
template <>
struct GaugeAdjustedBoundaryValue<Tags::BondiBeta> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::BondiBeta>, Tags::GaugeOmega,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& cauchy_gauge_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Spectral::Swsh::SwshInterpolator& interpolator) noexcept;
};

/*!
 * \brief Computes the evolution gauge quantity \f$\hat Q\f$ on the worldtube.
 *
 * \details The evolution gauge quantity \f$\hat Q\f$ obeys
 *
 * \f{align*}{
 * \hat Q =& \hat r^2 e^{-2 \hat \beta} (\hat K \partial_{\hat r} \hat U
 * + \hat J \partial_{\hat r} \hat{\bar U}),\\
 * \partial_{\hat r} \hat U
 * =& \frac{1}{2 \hat \omega^3}\left(\hat{\bar d} \partial_r U(\hat x^{\hat A})
 * - \hat c \partial_r \bar U(\hat x^{\hat A})\right)
 * + \frac{e^{2\hat \beta}}{\hat r^2 \hat \omega}
 * \left(\hat J \hat{\bar{\eth}} \hat \omega
 * - \hat K \hat \eth \hat \omega\right)
 * \left(-1 + \partial_{\hat y} \hat{\bar{J}} \partial_{\hat y} \hat J
 * - \left[\frac{\partial_{\hat y}(\hat J \hat{\bar{J}})}
 * {2 \hat K}\right]^2\right) \notag \\
 * & + 2 \frac{e^{2 \hat \beta}}{\hat \omega \hat r^2}
 * \left[ \hat{\bar{\eth}} \hat \omega \partial_{\hat y} \hat J
 * + \hat{\eth} \hat \omega \left(-\frac{\hat J \partial_{\hat y}
 * \hat{\bar J}
 * + \hat{\bar J} \partial_{\hat y} \hat J}{2 \hat K}\right) \right].
 * \f}
 *
 * where the explicit argument \f$\hat x^{\hat A}\f$ on the right-hand side
 * implies the need for an interpolation operation, and
 * \f$K = \sqrt{1 + J \bar J}\f$.
 */
template <>
struct GaugeAdjustedBoundaryValue<Tags::BondiQ> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiQ>>;

  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>, Tags::BondiJ,
      Tags::Dy<Tags::BondiJ>, Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>, Tags::GaugeC,
      Tags::GaugeD, Tags::GaugeOmega,
      Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          evolution_gauge_q,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& cauchy_gauge_dr_u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& volume_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& volume_dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const Spectral::Swsh::SwshInterpolator& interpolator,
      const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*evolution_gauge_q)), get(cauchy_gauge_dr_u),
               get(volume_j), get(volume_dy_j), get(evolution_gauge_r),
               get(evolution_gauge_beta), get(gauge_c), get(gauge_d),
               get(omega), get(eth_omega), interpolator, l_max);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> evolution_gauge_q,
      const SpinWeighted<ComplexDataVector, 1>& cauchy_gauge_dr_u,
      const SpinWeighted<ComplexDataVector, 2>& volume_j,
      const SpinWeighted<ComplexDataVector, 2>& volume_dy_j,
      const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_r,
      const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_beta,
      const SpinWeighted<ComplexDataVector, 2>& gauge_c,
      const SpinWeighted<ComplexDataVector, 0>& gauge_d,
      const SpinWeighted<ComplexDataVector, 0>& omega,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega,
      const Spectral::Swsh::SwshInterpolator& interpolator,
      size_t l_max) noexcept;
};

/*!
 * \brief Computes the evolution gauge quantity \f$\mathcal U\f$ on the
 * worldtube.
 *
 * \details Note that the boundary quantity computed by this function is, by
 * necessity, NOT the evolution gauge bondi \f$\hat U\f$, because there is
 * insufficient information at the point in the computation this will be
 * evaluated to completely determine \f$\hat{U}\f$. Instead, this determines
 * the boundary value of \f$\mathcal U\f$, which satisfies,
 *
 * \f{align*}{
 * \mathcal{U} - \mathcal{U}^{(0)} = \hat U,
 * \f}
 *
 * where the superscript \f$(0)\f$ denotes evaluation at \f$\mathcal I^+\f$. In
 * particular, the result of this computation may be used with the same
 * hypersurface equations as the full evolution gauge \f$\hat U\f$, because they
 * satisfy the same radial differential equation.
 *
 * \f$\mathcal U\f$ is computed by,
 *
 * \f{align*}{
 * \mathcal U = \frac{1}{2\hat \omega^2} \left(\hat{\bar d} U(\hat x^{\hat A})
 * - \hat c \bar U(\hat x^{\hat A}) \right)
 * - \frac{e^{2 \hat \beta}}{\hat r \hat \omega}
 * \left(\hat K \hat \eth \hat \omega
 * -  \hat J\hat{\bar{\eth}} \hat \omega\right),
 * \f}
 *
 * where the explicit argument \f$\hat x^{\hat A}\f$ on the right-hand side
 * implies the need for an interpolation operation, and
 * \f$K = \sqrt{1 + J \bar J}\f$.
 */
template <>
struct GaugeAdjustedBoundaryValue<Tags::BondiU> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiU>>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::BondiU>, Tags::BondiJ,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>, Tags::GaugeC,
      Tags::GaugeD, Tags::GaugeOmega,
      Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::LMax>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          evolution_gauge_u,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& cauchy_gauge_u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& volume_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const Spectral::Swsh::SwshInterpolator& interpolator,
      size_t l_max) noexcept;
};

/*!
 * \brief Computes the evolution gauge quantity \f$\hat W\f$ on the worldtube.
 *
 * \details The evolution gauge value \f$\hat W\f$ obeys
 *
 * \f{align*}{
 * \hat W =& W(\hat x^{\hat A}) + (\hat \omega - 1) / \hat r
 * + \frac{e^{2 \hat \beta}}{2 \hat \omega^2 \hat r}
 * \left(\hat J \left(\hat{\bar \eth} \hat \omega\right)^2
 * + \hat{\bar{J}} \left(\hat \eth \hat \omega\right) ^2
 * - 2 K \left( \hat \eth \hat \omega\right) \left(\hat{\bar \eth} \hat
 * \omega\right) \right)
 * - \frac{2 \partial_{u} \hat \omega}{\hat \omega}
 * - \frac{ \hat U \bar \eth \hat \omega + \hat{\bar U} \eth \hat \omega }
 * {\hat \omega},
 * \f}
 *
 * where the explicit argument \f$\hat x^{\hat A}\f$ on the right-hand side
 * implies the need for an interpolation operation and
 * \f$K = \sqrt{1 + J \bar J}\f$.
 */
template <>
struct GaugeAdjustedBoundaryValue<Tags::BondiW> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiW>>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::BondiW>, Tags::BondiJ,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiU>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>, Tags::BondiUAtScri,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>, Tags::GaugeOmega,
      Tags::Du<Tags::GaugeOmega>,
      Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& cauchy_gauge_w,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& volume_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& evolution_gauge_u,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&
          evolution_gauge_u_at_scri,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const Spectral::Swsh::SwshInterpolator& interpolator,
      const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*evolution_gauge_w)), get(cauchy_gauge_w),
               get(volume_j), get(evolution_gauge_u), get(evolution_gauge_beta),
               get(evolution_gauge_u_at_scri), get(evolution_gauge_r),
               get(omega), get(du_omega), get(eth_omega), interpolator, l_max);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> evolution_gauge_w,
      const SpinWeighted<ComplexDataVector, 0>& cauchy_gauge_w,
      const SpinWeighted<ComplexDataVector, 2>& volume_j,
      const SpinWeighted<ComplexDataVector, 1>& evolution_gauge_u,
      const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_beta,
      const SpinWeighted<ComplexDataVector, 1>& evolution_gauge_u_at_scri,
      const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_r,
      const SpinWeighted<ComplexDataVector, 0>& omega,
      const SpinWeighted<ComplexDataVector, 0>& du_omega,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega,
      const Spectral::Swsh::SwshInterpolator& interpolator,
      size_t l_max) noexcept;
};

/*!
 * \brief Computes the evolution gauge quantity \f$\hat H\f$ on the worldtube.
 *
 * \details The evolution gauge \f$\hat H\f$ obeys
 *
 * \f{align*}{
 *   \hat H =&
 * \frac{1}{2} \left(\mathcal{U}^{(0)} \hat{\bar \eth} \hat J
 * + \bar{\mathcal{U}}^{(0)} \hat{\eth} \hat J\right)
 * + \frac{\partial_{\hat u} \hat \omega
 * - \tfrac{1}{2} \left(\mathcal{U}^{(0)} \bar{\hat \eth}\hat \omega
 * + \bar{\mathcal{U}}^{(0)} \hat \eth \hat \omega \right) }{\hat \omega}
 * \left(2 \hat J - 2 \partial_{\hat y} \hat J\right)
 * - \hat J\hat{\bar \eth} \mathcal{U}^{(0)}
 * + \hat K \hat \eth \bar{\mathcal{U}}^{(0)}  \notag\\
 * &+ \frac{1}{4 \hat \omega^2} \left(\hat{\bar d}^2 H(\hat x^{\hat A})
 * + \hat c^2 \bar H(\hat x^{\hat A})
 * + \hat{\bar d} \hat c \frac{H(\hat x^{\hat A}) \bar J(\hat x^{\hat A})
 * + J(\hat x^{\hat A}) \bar H(\hat x^{\hat A})}{K}\right)
 * + 2 \frac{\partial_u R}{R} \partial_{\hat y} J
 * \f}
 *
 * where the superscript \f$(0)\f$ denotes evaluation at \f$\mathcal I^+\f$ and
 * the explicit \f$\hat x^{\hat A}\f$ arguments on the right-hand side imply
 * interpolation operations, and \f$K = \sqrt{1 + J \bar J}\f$,
 * \f$\hat K = \sqrt{1 + \hat J \hat{\bar J}}\f$.
 */
template <>
struct GaugeAdjustedBoundaryValue<Tags::BondiH> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiH>>;
  using argument_tags = tmpl::list<
      Tags::BondiJ, Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>,
      Tags::Dy<Tags::BondiJ>, Tags::BondiUAtScri,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>, Tags::GaugeC,
      Tags::GaugeD, Tags::GaugeOmega, Tags::Du<Tags::GaugeOmega>,
      Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          evolution_gauge_h,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& volume_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& cauchy_gauge_du_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& volume_dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&
          evolution_gauge_u_at_scri,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_r,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&
          evolution_gauge_du_r_divided_by_r,
      const Spectral::Swsh::SwshInterpolator& interpolator,
      const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*evolution_gauge_h)), get(volume_j),
               get(cauchy_gauge_du_j), get(volume_dy_j),
               get(evolution_gauge_u_at_scri), get(evolution_gauge_r),
               get(gauge_c), get(gauge_d), get(omega), get(du_omega),
               get(eth_omega), get(evolution_gauge_du_r_divided_by_r),
               interpolator, l_max);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> evolution_gauge_h,
      const SpinWeighted<ComplexDataVector, 2>& volume_j,
      const SpinWeighted<ComplexDataVector, 2>& cauchy_gauge_du_j,
      const SpinWeighted<ComplexDataVector, 2>& volume_dy_j,
      const SpinWeighted<ComplexDataVector, 1>& evolution_gauge_u_at_scri,
      const SpinWeighted<ComplexDataVector, 0>& evolution_gauge_r,
      const SpinWeighted<ComplexDataVector, 2>& gauge_c,
      const SpinWeighted<ComplexDataVector, 0>& gauge_d,
      const SpinWeighted<ComplexDataVector, 0>& omega,
      const SpinWeighted<ComplexDataVector, 0>& du_omega,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega,
      const SpinWeighted<ComplexDataVector, 0>&
          evolution_gauge_du_r_divided_by_r,
      const Spectral::Swsh::SwshInterpolator& interpolator,
      size_t l_max) noexcept;
};

/*!
 * \brief Update the Cauchy gauge cartesian coordinate derivative \f$\partial_u
 * x(\hat x)\f$, as well as remaining gauge quantities \f$\mathcal U^{(0)}\f$,
 * \f$\hat U \equiv \mathcal U - \mathcal U^{(0)}\f$, and \f$\partial_{\hat u}
 * \hat \omega\f$ to maintain asymptotically inertial angular coordinates.
 *
 * \details The constraint we must satisfy to maintain the asymptotically
 * inertial angular coordinates is
 *
 * \f{align*}{
 * \partial_{\hat u} x^A =  \mathcal U^{(0) \hat A} \partial_{\hat A} x^{A},
 * \f}
 *
 * which we compute for a representative Cartesian coordinate set on the unit
 * sphere, to maintain representability and ensure that angular transform and
 * derivative operations keep the desired precision. The equation we use for the
 * Cartesian analog is:
 *
 * \f{align*}{
 * \partial_{\hat u} x^i &= \frac{1}{2} (\bar{\mathcal U}^{(0)} \hat \eth x^i +
 * \mathcal U^{(0)} \hat{\bar \eth} x^i ) \\
 * &= \text{Re}(\bar{\mathcal U}^{(0)} \hat \eth x^i)
 * \f}
 *
 * This computation completes the unfixed degrees of freedom for the coordinate
 * transformation at the boundary, so also computes the gauge quantities that
 * rely on this information \f$\mathcal U^{(0)}\f$,
 * \f$\hat U\f$, and \f$\partial_{\hat u} \hat \omega\f$.
 *
 * The time derivative of \f$\hat \omega\f$ is calculated from the equation
 * \f{align*}{
 * \partial_{\hat u} \hat \omega
 * = \frac{\hat \omega}{4} (\hat{\bar \eth} \mathcal U^{(0)}
 * + \hat \eth \bar{\mathcal U}^{(0)})
 * + \frac{1}{2} (\mathcal U^{(0)} \hat{\bar \eth} \hat \omega
 * + \bar{\mathcal U}^{(0)} \hat \eth \hat \omega)
 * \f}
 * \warning Before this update call the quantity stored in the tag
 * `Cce::Tags::BondiU` represents \f$\mathcal U\f$, and after this update call,
 * it represents \f$\hat U\f$ (the true evolution gauge quantity).
 */
struct GaugeUpdateTimeDerivatives {
  using return_tags =
      tmpl::list<::Tags::dt<Tags::CauchyCartesianCoords>, Tags::BondiUAtScri,
                 Tags::BondiU, Tags::Du<Tags::GaugeOmega>>;
  using argument_tags =
      tmpl::list<Tags::CauchyCartesianCoords, Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::LMax>;

  static void apply(
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_du_x,
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          evolution_gauge_u_at_scri,
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> volume_u,
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_omega,
      const tnsr::i<DataVector, 3>& cartesian_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      size_t l_max) noexcept;
};

/*!
 * \brief Update the angular coordinates stored in `AngularTag` via
 * trigonometric operations applied to the Cartesian coordinates stored in
 * `CartesianTag`.
 *
 * \details This function also normalizes the Cartesian coordinates stored in
 * `CartesianTag`, which is the desired behavior for the CCE boundary
 * computation.
 */
template <typename AngularTag, typename CartesianTag>
struct GaugeUpdateAngularFromCartesian {
  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<AngularTag, CartesianTag>;

  static void apply(
      const gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_coordinates,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
          cartesian_coordinates) noexcept {
    // normalize the cartesian coordinates
    const DataVector one_over_cartesian_r =
        1.0 / sqrt(square(get<0>(*cartesian_coordinates)) +
                   square(get<1>(*cartesian_coordinates)) +
                   square(get<2>(*cartesian_coordinates)));

    get<0>(*cartesian_coordinates) *= one_over_cartesian_r;
    get<1>(*cartesian_coordinates) *= one_over_cartesian_r;
    get<2>(*cartesian_coordinates) *= one_over_cartesian_r;

    const auto& x = get<0>(*cartesian_coordinates);
    const auto& y = get<1>(*cartesian_coordinates);
    const auto& z = get<2>(*cartesian_coordinates);

    get<0>(*angular_coordinates) = atan2(sqrt(square(x) + square(y)), z);
    get<1>(*angular_coordinates) = atan2(y, x);
  }
};

namespace detail {
void gauge_update_jacobian_from_coordinates_apply_impl(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        gauge_factor_spin_2,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        gauge_factor_spin_0,
    gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_source_coordinates,
    const tnsr::i<DataVector, 3>& cartesian_source_coordinates,
    size_t l_max) noexcept;
}  // namespace detail

/*!
 * \brief From the angular coordinates `AngularCoordinateTag` and the Cartesian
 * coordinates `CartesianCoordinateTag`, determine the spin-weighted Jacobian
 * factors `GaugeFactorSpin2` and `GaugeFactorSpin0`.
 *
 * \details This is most often used in the context of generating the Jacobians
 * in the evolution-gauge coordinates from the Cauchy collocation points as a
 * function of the evolution gauge coordinates. In this concrete case, the
 * `GaugeFactorSpin2` is the gauge factor \f$\hat c\f$ and takes the value
 *
 * \f{align*}{
 * \hat c = \hat q^{\hat A} \partial_{\hat A}(x^A) q_A,
 * \f}
 *
 * and the `GaugeFactorSpin0` is the gauge factor \f$\hat d\f$ and takes the
 * value
 *
 * \f{align*}{
 * \hat d = \hat{\bar q}^{\hat A} \partial_{\hat A}(x^A) q_A.
 * \f}
 *
 * The more generic template construction is employed so that the spin-weighted
 * Jacobians can also be computed between two arbitrary gauges, including the
 * inverse Jacobians associated with moving from the evolution gauge to the
 * Cauchy gauge.
 */
template <typename GaugeFactorSpin2, typename GaugeFactorSpin0,
          typename AngularCoordinateTag, typename CartesianCoordinateTag>
struct GaugeUpdateJacobianFromCoordinates {
  using return_tags =
      tmpl::list<GaugeFactorSpin2, GaugeFactorSpin0, AngularCoordinateTag>;
  using argument_tags =
      tmpl::list<CartesianCoordinateTag, Spectral::Swsh::Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          gauge_factor_spin_2,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          gauge_factor_spin_0,
      const gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_source_coordinates,
      const tnsr::i<DataVector, 3>& cartesian_source_coordinates,
      const size_t l_max) noexcept {
    detail::gauge_update_jacobian_from_coordinates_apply_impl(
        gauge_factor_spin_2, gauge_factor_spin_0, angular_source_coordinates,
        cartesian_source_coordinates, l_max);
  }
};

/*!
 * \brief Update the interpolator stored in
 * `Spectral::Swsh::Tags::SwshInterpolator<AngularCoordinates>`.
 *
 * \details Note that the `AngularCoordinates` associated with the interpolator
 * should be the source coordinates. For instance, when interpolating a quantity
 * defined on the Cauchy gauge collocation points to the evolution gauge
 * collocation points, the interpolator input should be the Cauchy coordinates
 * points as a function of the evolution gauge coordinates (at the evolution
 * gauge collocation points).
 */
template <typename AngularCoordinates>
struct GaugeUpdateInterpolator {
  using return_tags =
      tmpl::list<Spectral::Swsh::Tags::SwshInterpolator<AngularCoordinates>>;
  using argument_tags =
      tmpl::list<AngularCoordinates, Spectral::Swsh::Tags::LMax>;

  static void apply(
      const gsl::not_null<Spectral::Swsh::SwshInterpolator*> interpolator,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
          angular_coordinates,
      const size_t l_max) noexcept {
    // throw away the old interpolator and generate a new one for the current
    // grid points.
    *interpolator = Spectral::Swsh::SwshInterpolator(
        get<0>(angular_coordinates), get<1>(angular_coordinates), l_max);
  }
};

/*!
 * \brief Update the quantity \f$\hat \omega\f$ and \f$\hat \eth \hat \omega\f$
 * for updated spin-weighted Jacobian quantities \f$\hat c\f$ and \f$\hat d\f$.
 *
 * \details The conformal factor \f$\hat \omega\f$ can be determined by the
 * angular determinant from the spin-weighted Jacobian factors as
 *
 * \f{align*}{
 * \hat \omega = \frac{1}{2} \sqrt{\hat d \hat{\bar d} - \hat c \hat{\bar c}}.
 * \f}
 */
struct GaugeUpdateOmega {
  using argument_tags =
      tmpl::list<Tags::GaugeC, Tags::GaugeD, Spectral::Swsh::Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> omega,
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
      size_t l_max) noexcept;
};

/*!
 * \brief Initialize to default values (identity transform) all of the angular
 * gauge quantities for the boundary gauge transforms.
 *
 * \details The updated quantities are the Cauchy angular and Cartesian
 * coordinates, as well as the spin-weighted gauge factors and the conformal
 * factor. All quantities are initialized to the appropriate value for the
 * identity transform of angular coordinates. Using this initialization function
 * ensures that the evolution gauge and the Cauchy gauge angular coordinates
 * agree on the first evaluated time.
 * - `CauchyAngularCoords` are set to the angular collocation values for the
 * spin-weighted spherical harmonic library
 * - `CauchyCartesianCoords` are set to the Cartesian coordinates for the
 * `CauchyAngularCoords` evaluated on a unit sphere.
 * - `GaugeC` is set to 0
 * - `GaugeD` is set to 2
 * - `GaugeOmega` is set to 1
 */
struct InitializeGauge {
  using return_tags =
      tmpl::list<Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords,
                 Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmega>;
  using argument_tags = tmpl::list<Spectral::Swsh::Tags::LMax>;

  static void apply(
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> gauge_c,
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> gauge_d,
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> omega,
      size_t l_max) noexcept;
};
}  // namespace Cce
