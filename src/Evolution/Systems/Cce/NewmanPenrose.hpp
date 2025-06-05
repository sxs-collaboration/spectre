// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {

/// \cond
namespace Tags {
struct LMax;
}  // namespace Tags
template <typename Tag>
struct VolumeWeyl;
/// \endcond

/*!
 * \brief Compute the (adapted) Newman-Penrose spin coefficient
 * $\alpha^{SW}$ in the volume, in the conventions of \cite Moxon2020gha .
 *
 * \details The definition of $\alpha$ is:
 *
 * \begin{align}
 * \alpha = \frac{1}{2}\left( \bar{m}^\mu \bar{m}^\nu \nabla_\nu m_\mu
 *   - n^\mu \bar{m}^\nu \nabla_\nu l_\mu \right) .
 * \end{align}
 *
 * It does not have a well-defined spin weight; but by using a reference
 * spin-connection, the adapted spin coefficient $\alpha^{SW} \equiv \alpha -
 * \alpha_{ref}$ does have a well-defined spin weight of -1. In the language
 * of \cite Moxon2020gha, that is equivalent to setting $\Theta=0$. This
 * gives the expression
 *
 * \begin{align}
 *   \alpha^{SW} = \frac{1-y}{32R} \Bigg\{
 *     & \frac{1}{\sqrt{1+K}} \Bigg[
 *       \frac{\bar{J}^2\eth J}{K(1+K)}
 *       +\frac{\bar\eth(J\bar{J}) - \eth\bar{J}}{K} \\ \nonumber
 *      & {}+ \left( 2\bar{J}(Q+2\eth\beta)-3\eth\bar{J}\right)
 *     \Bigg]
 *     -2\sqrt{1+K}\left(\overline{Q+2\eth\beta}\right)
 *   \Bigg\}
 * \end{align}
 */
void newman_penrose_alpha(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> np_alpha,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 3>>& eth_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);

/*!
 * \brief Compute the (adapted) Newman-Penrose spin coefficient
 * $\beta_{NP}^{SW}$ in the volume, in the conventions of
 * \cite Moxon2020gha .
 *
 * \details The definition of $\beta_{NP}$ is:
 *
 * \begin{align}
 * \beta_{NP} = \frac{1}{2}\left( \bar{m}^\mu m^\nu \nabla_\nu m_\mu
 *   - n^\mu m^\nu \nabla_\nu l_\mu \right) .
 * \end{align}
 *
 * It does not have a well-defined spin weight; but by using a reference
 * spin-connection, the adapted spin coefficient $\beta_{NP}^{SW} \equiv
 * \beta_{NP} - \beta_{NP,ref}$ does have a well-defined spin weight of +1. In
 * the language of \cite Moxon2020gha, that is equivalent to setting
 * $\Theta=0$. This gives the expression
 *
 * \begin{align}
 *   \beta_{NP}^{SW} = \frac{1-y}{32R} \Bigg\{
 *     & \frac{1}{\sqrt{1+K}} \Bigg[
 *       -\frac{J^2\bar\eth\bar J}{K(1+K)}
 *       +\frac{-\eth(J\bar{J}) + \bar\eth J}{K} \\ \nonumber
 *      & {}+ \left( 2J(\overline{Q+2\eth\beta})+3\bar\eth J\right)
 *     \Bigg]
 *     -2\sqrt{1+K}\left(Q+2\eth\beta\right)
 *   \Bigg\}
 * \end{align}
 */
void newman_penrose_beta(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, +1>>*> np_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 3>>& eth_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);

/*!
 * \brief Compute the (adapted) Newman-Penrose spin coefficient
 * $\gamma^{SW}$ in the volume, in the conventions of
 * \cite Moxon2020gha .
 *
 * \details The definition of $\gamma$ is:
 *
 * \begin{align}
 * \gamma = \frac{1}{2}\left( \bar{m}^\mu n^\nu \nabla_\nu m_\mu
 *   - n^\mu n^\nu \nabla_\nu l_\mu \right) .
 * \end{align}
 *
 * It does not have a well-defined spin weight; but by using a reference
 * spin-connection, the adapted spin coefficient $\gamma^{SW} \equiv
 * \gamma - \gamma_{ref}$ does have a well-defined spin weight of 0. In
 * the language of \cite Moxon2020gha, that is equivalent to setting
 * $\Theta=0$. This gives the expression
 *
 * \begin{align}
 *   \gamma^{SW} = \frac{e^{-2\beta}}{4\sqrt{2}} \Bigg\{
 *        & \frac{1}{2(1+K)} \Bigg[
 *          (1-y)\left(\frac{1-y}{2R}+W\right) (\bar{J}\partial_y J
 *                     - J\partial_y\bar{J}) \\ \nonumber
 *       & + \left( 2\bar{H} J - 2H\bar{J} + U J \overline{\eth J}
 *                  - U \bar{J} \bar{\eth}J
 *        + \bar{U} J \eth \bar{J} - \bar{U} \bar{J}\eth J \right)
 *        \Bigg] \\ \nonumber
 *      & + 2(1-y) \partial_y W \\ \nonumber
 *      & + \left(2W+J\overline{\eth U}-\bar{J}\eth U
 *                -K\eth\bar{U}+K\bar{\eth}U\right)
 *      \Bigg\}
 * \end{align}
 */
void newman_penrose_gamma(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> np_gamma,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 3>>& eth_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_h,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);

/*!
 * \brief Compute the (adapted) Newman-Penrose spin coefficient
 * $\epsilon^{SW}$ in the volume, in the conventions of
 * \cite Moxon2020gha .
 *
 * \details The definition of $\epsilon$ is:
 *
 * \begin{align}
 * \epsilon = \frac{1}{2}\left( \bar{m}^\mu l^\nu \nabla_\nu m_\mu
 *   - n^\mu l^\nu \nabla_\nu l_\mu \right) .
 * \end{align}
 *
 * It does not have a well-defined spin weight; but by using a reference
 * spin-connection, the adapted spin coefficient $\epsilon^{SW} \equiv
 * \epsilon - \epsilon_{ref}$ does have a well-defined spin weight of 0. In
 * the language of \cite Moxon2020gha, that is equivalent to setting
 * $\Theta=0$. This gives the expression
 *
 * \begin{align}
 *   \epsilon^{SW} = \frac{(1-y)^2}{2\sqrt{2} R} \left(
 *     \partial_y \beta + \frac{J\partial_y \bar{J}
 *         - \bar{J}\partial_y J}{8(1+K)}
 *     \right)
 * \end{align}
 */
void newman_penrose_epsilon(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> np_epsilon,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);

// We do not define newman_penrose_kappa since in our choice of tetrad, it's 0

/*!
 * \brief Compute the Newman-Penrose spin coefficient
 * $\tau$ in the volume, in the conventions of
 * \cite Moxon2020gha .
 *
 * \details The definition of $\tau$ is:
 *
 * \begin{align}
 * \tau = - m^\mu n^\nu \nabla_\nu l_\mu .
 * \end{align}
 *
 * It has a spin weight of +1. This gives the expression
 *
 * \begin{align}
 *   \tau = \frac{1-y}{8 R} \left(
 *     \sqrt{1+K} (2\eth\beta - Q)
 *     - \frac{J (\overline{2\eth\beta-Q})}{\sqrt{1+K}}
 *     \right)
 * \end{align}
 */
void newman_penrose_tau(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, +1>>*> np_tau,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);

/*!
 * \brief Compute the Newman-Penrose spin coefficient
 * $\sigma$ in the volume, in the conventions of
 * \cite Moxon2020gha .
 *
 * \details The definition of $\sigma$ is:
 *
 * \begin{align}
 * \sigma = - m^\mu m^\nu \nabla_\nu l_\mu .
 * \end{align}
 *
 * It has a spin weight of +2. This gives the expression
 *
 * \begin{align}
 *   \sigma = \frac{(1-y)^2}{8 \sqrt{2} K R} \left(
 *     \frac{J^2 \partial_y \bar{J}}{1+K}
 *     - (1+K)\partial_y J
 *     \right)
 * \end{align}
 */
void newman_penrose_sigma(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, +2>>*> np_sigma,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);

/*!
 * \brief Compute the Newman-Penrose spin coefficient
 * $\rho$ in the volume, in the conventions of
 * \cite Moxon2020gha .
 *
 * \details The definition of $\rho$ is:
 *
 * \begin{align}
 * \rho = - m^\mu \bar{m}^\nu \nabla_\nu l_\mu .
 * \end{align}
 *
 * It has a spin weight of 0. This gives the expression
 *
 * \begin{align}
 *   \rho = - \frac{(1-y)}{2 \sqrt{2} R}
 * \end{align}
 */
void newman_penrose_rho(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> np_rho,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);

/*!
 * \brief Compute the Newman-Penrose spin coefficient
 * $\pi$ in the volume, in the conventions of
 * \cite Moxon2020gha .
 *
 * \details The definition of $\pi$ is:
 *
 * \begin{align}
 * \pi = \bar{m}^\mu l^\nu \nabla_\nu n_\mu .
 * \end{align}
 *
 * It has a spin weight of -1. This gives the expression
 *
 * \begin{align}
 *   \pi = \frac{(1-y)}{8 R} \left[
 *     \frac{\bar{J}(Q+2\eth\beta)}{\sqrt{1+K}}
 *     - \sqrt{1+K} (\overline{Q+2\eth\beta})
 *   \right]
 * \end{align}
 */
void newman_penrose_pi(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> np_pi,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_q,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);

/*!
 * \brief Compute the Newman-Penrose spin coefficient
 * $\nu$ in the volume, in the conventions of
 * \cite Moxon2020gha .
 *
 * \details The definition of $\nu$ is:
 *
 * \begin{align}
 * \nu = \bar{m}^\mu n^\nu \nabla_\nu n_\mu .
 * \end{align}
 *
 * It has a spin weight of -1. This gives the expression
 *
 * \begin{align}
 *   \nu = \frac{e^{-2\beta}}{2} \left[
 *     \frac{\bar{J}\eth W}{\sqrt{1+K}}
 *     - \sqrt{1+K} \bar{\eth}W
 *   \right]
 * \end{align}
 */
void newman_penrose_nu(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> np_nu,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta);

/*!
 * \brief Compute the Newman-Penrose spin coefficient
 * $\mu$ in the volume, in the conventions of
 * \cite Moxon2020gha .
 *
 * \details The definition of $\mu$ is:
 *
 * \begin{align}
 * \mu = \bar{m}^\mu m^\nu \nabla_\nu n_\mu .
 * \end{align}
 *
 * It has a spin weight of 0. This gives the expression
 *
 * \begin{align}
 *   \mu = \frac{e^{-2\beta}}{2\sqrt{2}} \left[
 *     \eth\bar{U} + \bar{\eth}U - \frac{1-y}{R} - 2W
 *   \right]
 * \end{align}
 */
void newman_penrose_mu(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> np_mu,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);

/*!
 * \brief Compute the Newman-Penrose spin coefficient
 * $\lambda$ in the volume, in the conventions of
 * \cite Moxon2020gha .
 *
 * \details The definition of $\lambda$ is:
 *
 * \begin{align}
 * \lambda = \bar{m}^\mu \bar{m}^\nu \nabla_\nu n_\mu .
 * \end{align}
 *
 * It has a spin weight of -2. This gives the expression
 *
 * \begin{align}
 *   \lambda = \frac{e^{-2\beta}}{4\sqrt{2}} \Bigg\{ &
 *       \left[\frac{1-y}{R}+2W\right] \frac{1-y}{2(1+K)}
 *       \left[ \frac{\bar{J}^2\partial_y J - \partial_y \bar{J}}{K} -
 *         (2+K) \partial_y\bar{J} \right] \\ \nonumber
 *       &{}+ 2(1+K)\bar{\eth}\bar{U}  + \frac{\bar{I}}{K}
 *       +\left[ \bar{I} +2\bar{J}(\bar\eth U - \eth\bar U) \right]
 *       - \frac{\bar{J}^2}{K(1+K)}(I + 2K\eth U)
 *     \Bigg\}
 * \end{align}
 *
 * where we have defined the temporary quantity
 *
 * \begin{align}
 *   I = 2H+U\bar\eth{J}+\bar{U}\eth J
 * \end{align}
 *
 */
void newman_penrose_lambda(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> np_lambda,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 3>>& eth_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_h,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);

namespace Tags {
/*!
 * \brief Compute tag for $\alpha^{SW}$ in the volume.
 *
 * \details See documentation of `newman_penrose_alpha()` for definition.
 */
struct NewmanPenroseAlphaCompute : Tags::NewmanPenroseAlpha, db::ComputeTag {
  using base = Tags::NewmanPenroseAlpha;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::BondiJ,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Tags::BondiK, Tags::BondiR, Tags::BondiQ,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::OneMinusY>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 3>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_alpha);
};

/*!
 * \brief Compute tag for $\beta_{NP}^{SW}$ in the volume.
 *
 * \details See documentation of `newman_penrose_beta()` for definition.
 */
struct NewmanPenroseBetaCompute : Tags::NewmanPenroseBeta, db::ComputeTag {
  using base = Tags::NewmanPenroseBeta;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::BondiJ,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Tags::BondiK, Tags::BondiR, Tags::BondiQ,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::OneMinusY>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, +1>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 3>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_beta);
};

/*!
 * \brief Compute tag for $\gamma^{SW}$ in the volume.
 *
 * \details See documentation of `newman_penrose_gamma()` for definition.
 */
struct NewmanPenroseGammaCompute : Tags::NewmanPenroseGamma, db::ComputeTag {
  using base = Tags::NewmanPenroseGamma;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::BondiJ, Tags::Dy<Tags::BondiJ>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Tags::BondiK, Tags::BondiH, Tags::BondiR,
                 Tags::BondiU,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Tags::BondiW, Tags::Dy<Tags::BondiW>,
                 Tags::Exp2Beta, Tags::OneMinusY>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 3>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_gamma);
};

/*!
 * \brief Compute tag for $\epsilon^{SW}$ in the volume.
 *
 * \details See documentation of `newman_penrose_epsilon()` for definition.
 */
struct NewmanPenroseEpsilonCompute :
      Tags::NewmanPenroseEpsilon, db::ComputeTag {
  using base = Tags::NewmanPenroseEpsilon;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::BondiJ, Tags::Dy<Tags::BondiJ>,
                 Tags::BondiK, Tags::BondiR,
                 Tags::Dy<Tags::BondiBeta>, Tags::OneMinusY>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_epsilon);
};

// We do not implement a compute tag for NewmanPenroseKappa, since it vanishes

/*!
 * \brief Compute tag for $\tau$ in the volume.
 *
 * \details See documentation of `newman_penrose_tau()` for definition.
 */
struct NewmanPenroseTauCompute : Tags::NewmanPenroseTau, db::ComputeTag {
  using base = Tags::NewmanPenroseTau;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::BondiJ, Tags::BondiK,
                 Tags::BondiR, Tags::BondiQ,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::OneMinusY>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, +1>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_tau);
};

/*!
 * \brief Compute tag for $\sigma$ in the volume.
 *
 * \details See documentation of `newman_penrose_sigma()` for definition.
 */
struct NewmanPenroseSigmaCompute : Tags::NewmanPenroseSigma, db::ComputeTag {
  using base = Tags::NewmanPenroseSigma;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::BondiJ, Tags::Dy<Tags::BondiJ>,
                 Tags::BondiK, Tags::BondiR, Tags::OneMinusY>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, +2>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_sigma);
};

/*!
 * \brief Compute tag for $\rho$ in the volume.
 *
 * \details See documentation of `newman_penrose_rho()` for definition.
 */
struct NewmanPenroseRhoCompute : Tags::NewmanPenroseRho, db::ComputeTag {
  using base = Tags::NewmanPenroseRho;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<Tags::BondiR, Tags::OneMinusY>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_rho);
};

/*!
 * \brief Compute tag for $\pi$ in the volume.
 *
 * \details See documentation of `newman_penrose_pi()` for definition.
 */
struct NewmanPenrosePiCompute : Tags::NewmanPenrosePi, db::ComputeTag {
  using base = Tags::NewmanPenrosePi;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::BondiJ, Tags::BondiK,
                 Tags::BondiR, Tags::BondiQ,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::OneMinusY>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_pi);
};

/*!
 * \brief Compute tag for $\nu$ in the volume.
 *
 * \details See documentation of `newman_penrose_nu()` for definition.
 */
struct NewmanPenroseNuCompute : Tags::NewmanPenroseNu, db::ComputeTag {
  using base = Tags::NewmanPenroseNu;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::BondiJ, Tags::BondiK,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiW,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::Exp2Beta>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_nu);
};

/*!
 * \brief Compute tag for $\mu$ in the volume.
 *
 * \details See documentation of `newman_penrose_mu()` for definition.
 */
struct NewmanPenroseMuCompute : Tags::NewmanPenroseMu, db::ComputeTag {
  using base = Tags::NewmanPenroseMu;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::BondiR, Tags::BondiW,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Tags::Exp2Beta, Tags::OneMinusY>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_mu);
};

/*!
 * \brief Compute tag for $\lambda$ in the volume.
 *
 * \details See documentation of `newman_penrose_lambda()` for definition.
 */
struct NewmanPenroseLambdaCompute : Tags::NewmanPenroseLambda, db::ComputeTag {
  using base = Tags::NewmanPenroseLambda;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::BondiJ, Tags::Dy<Tags::BondiJ>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Tags::BondiK, Tags::BondiH,
                 Tags::BondiR, Tags::BondiU,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Tags::BondiW,
                 Tags::Exp2Beta, Tags::OneMinusY>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 3>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &newman_penrose_lambda);
};

}  // namespace Tags

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

namespace Tags {
/*!
 * \brief Compute tag for $\Psi_0$ in the volume.
 *
 * \details Uses `apply` function from `VolumeWeyl<Tags::Psi0>` for the
 * computation.
 */
struct Psi0Compute : Tags::Psi0, db::ComputeTag {
  using base = Tags::Psi0;
  using return_type = typename base::type;
  using argument_tags = typename VolumeWeyl<Tags::Psi0>::argument_tags;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &VolumeWeyl<Tags::Psi0>::apply);
};
}  // namespace Tags

/*!
 * \brief Compute the Weyl scalar \f$\Psi_1\f$ in the volume according to the
 * standard set of Newman-Penrose vectors.
 *
 * \details Our convention is \f$\Psi_1 =
 * l^\alpha n^\beta l^\mu m^\nu C_{\alpha
 * \beta \mu \nu}\f$.
 *
 * \f{align*}{
 * \Psi_1 =
 * &\frac{(1-y)^2}{\sqrt{128} \sqrt{1 + K} R^2}\Bigg\{
 * J(\bar{\eth }\beta + \tfrac{1}{2} \bar{Q})
 * - (1 + K) (\eth \beta + \tfrac{1}{2} Q) \\
 * &+(1-y)\Bigg[
 * (1 + K)\eth\partial_{y}\beta - J\bar{\eth }\partial_{y}\beta
 * + \left(- J \frac{\bar{\eth} R}{R} + (1 + K) \frac{\eth R}{R}\right)
 * \partial_{y}\beta \\
 * &\quad + \frac{1}{4K}\Bigg(
 * J\left\{
 * -2 \partial_{y}\bar{Q} + \partial_{y}\bar{J} [2 (\eth\beta + \tfrac{1}{2}Q) +
 * J(\bar{\eth}\beta + \tfrac{1}{2} \bar{Q})]
 * \right\}\\
 * &\qquad +(1+K)\Big\{
 * 2 (\partial_{y}Q + J \partial_{y}\bar{Q}) + (\bar{J} \partial_{y}J - J
 * \partial_{y}\bar{J}) (\eth\beta + \tfrac{1}{2} Q) \\
 * &\qquad\quad - (1+K)
 * [2 \partial_{y}Q + \partial _{y}J (\bar{\eth }\beta + \tfrac{1}{2} \bar{Q})]
 * \Big\}\Bigg)\Bigg]\Bigg\}
 * \f}
 */
template <>
struct VolumeWeyl<Tags::Psi1> {
  using return_tags = tmpl::list<Tags::Psi1>;
  using argument_tags =
      tmpl::list<Tags::BondiJ, Tags::Dy<Tags::BondiJ>,
                 Tags::BondiK, Tags::BondiQ,
                 Tags::Dy<Tags::BondiQ>, Tags::BondiR, Tags::EthRDividedByR,
                 Tags::Dy<Tags::BondiBeta>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::OneMinusY>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> psi_1,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_q,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_q,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_dy_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);
};

namespace Tags {
/*!
 * \brief Compute tag for $\Psi_1$ in the volume.
 *
 * \details Uses `apply` function from `VolumeWeyl<Tags::Psi1>` for the
 * computation.
 */
struct Psi1Compute : Tags::Psi1, db::ComputeTag {
  using base = Tags::Psi1;
  using return_type = typename base::type;
  using argument_tags = typename VolumeWeyl<Tags::Psi1>::argument_tags;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&)>(
      &VolumeWeyl<Tags::Psi1>::apply);
};
}  // namespace Tags

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
