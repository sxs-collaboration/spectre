// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/SpinWeighted.hpp"  // IWYU pragma: keep
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Cce::Tags::BondiBeta
// IWYU pragma: no_forward_declare Cce::Tags::DuRDividedByR
// IWYU pragma: no_forward_declare Cce::Tags::EthRDividedByR
// IWYU pragma: no_forward_declare Cce::Tags::Exp2Beta
// IWYU pragma: no_forward_declare Cce::Tags::BondiH
// IWYU pragma: no_forward_declare Cce::Tags::BondiJ
// IWYU pragma: no_forward_declare Cce::Tags::BondiJbar
// IWYU pragma: no_forward_declare Cce::Tags::JbarQMinus2EthBeta
// IWYU pragma: no_forward_declare Cce::Tags::BondiK
// IWYU pragma: no_forward_declare Cce::Tags::OneMinusY
// IWYU pragma: no_forward_declare Cce::Tags::BondiQ
// IWYU pragma: no_forward_declare Cce::Tags::BondiR
// IWYU pragma: no_forward_declare Cce::Tags::BondiU
// IWYU pragma: no_forward_declare Cce::Tags::BondiUbar
// IWYU pragma: no_forward_declare Cce::Tags::BondiW
// IWYU pragma: no_forward_declare ::Tags::Multiplies
// IWYU pragma: no_forward_declare Cce::Tags::Dy
// IWYU pragma: no_forward_declare Cce::Tags::Integrand
// IWYU pragma: no_forward_declare Cce::Tags::LinearFactor
// IWYU pragma: no_forward_declare Cce::Tags::LinearFactorForConjugate
// IWYU pragma: no_forward_declare Cce::Tags::PoleOfIntegrand
// IWYU pragma: no_forward_declare Cce::Tags::RegularIntegrand
// IWYU pragma: no_forward_declare Spectral::Swsh::Tags::Eth
// IWYU pragma: no_forward_declare Spectral::Swsh::Tags::EthEth
// IWYU pragma: no_forward_declare Spectral::Swsh::Tags::EthEthbar
// IWYU pragma: no_forward_declare Spectral::Swsh::Tags::Ethbar
// IWYU pragma: no_forward_declare Spectral::Swsh::Tags::EthbarEthbar
// IWYU pragma: no_forward_declare Spectral::Swsh::Tags::Derivative
// IWYU pragma: no_forward_declare Tags::TempTensor
// IWYU pragma: no_forward_declare Tags::SpinWeighted
// IWYU pragma: no_forward_declare SpinWeighted
// IWYU pragma: no_forward_declare Tensor

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {

namespace detail {
template <typename BondiVariable>
struct integrand_terms_to_compute_for_bondi_variable_impl;

// template specializations for the individual tags
template <>
struct integrand_terms_to_compute_for_bondi_variable_impl<Tags::BondiBeta> {
  using type = tmpl::list<Tags::Integrand<Tags::BondiBeta>>;
};
template <>
struct integrand_terms_to_compute_for_bondi_variable_impl<Tags::BondiQ> {
  using type = tmpl::list<Tags::PoleOfIntegrand<Tags::BondiQ>,
                          Tags::RegularIntegrand<Tags::BondiQ>>;
};
template <>
struct integrand_terms_to_compute_for_bondi_variable_impl<Tags::BondiU> {
  using type = tmpl::list<Tags::Integrand<Tags::BondiU>>;
};
template <>
struct integrand_terms_to_compute_for_bondi_variable_impl<Tags::BondiW> {
  using type = tmpl::list<Tags::PoleOfIntegrand<Tags::BondiW>,
                          Tags::RegularIntegrand<Tags::BondiW>>;
};
template <>
struct integrand_terms_to_compute_for_bondi_variable_impl<Tags::BondiH> {
  using type = tmpl::list<Tags::PoleOfIntegrand<Tags::BondiH>,
                          Tags::RegularIntegrand<Tags::BondiH>,
                          Tags::LinearFactor<Tags::BondiH>,
                          Tags::LinearFactorForConjugate<Tags::BondiH>>;
};
}  // namespace detail

/// \brief A struct for providing a `tmpl::list` of integrand tags that need to
/// be computed before integration can proceed for a given Bondi variable tag.
template <typename BondiVariable>
using integrand_terms_to_compute_for_bondi_variable =
    typename detail::integrand_terms_to_compute_for_bondi_variable_impl<
        BondiVariable>::type;

/*!
 * \brief Computes one of the inputs for the integration of one of the
 * Characteristic hypersurface equations.
 *
 * \details The template argument must be one of the integrand-related prefix
 * tags templated on a Bondi quantity tag for which that integrand is required.
 * The relevant prefix tags are `Tags::Integrand`, `Tags::PoleOfIntegrand`,
 * `Tags::RegularIntegrand`, `Tags::LinearFactor`, and
 * `Tags::LinearFactorForConjugate`. The Bondi quantity tags that these tags may
 * wrap are `Tags::BondiBeta`, `Tags::BondiQ`, `Tags::BondiU`, `Tags::BondiW`,
 * and `Tags::BondiH`.
 *
 * The integrand terms which may be computed for a given Bondi variable are
 * enumerated in the type alias `integrand_terms_to_compute_for_bondi_variable`,
 * which takes as a single template argument the tag for which integrand terms
 * would be computed, and is a `tmpl::list` of the integrand terms needed.
 *
 * The resulting quantity is returned by `not_null` pointer, and the required
 * argument tags are given in `return_tags` and `argument_tags` type aliases,
 * where the `return_tags` are passed by `not_null` pointer (so include
 * temporary buffers) and the `argument_tags` are passed by const reference.
 *
 * Additional mathematical details for each of the computations can be found in
 * the template specializations of this struct. All of the specializations have
 * a static `apply` function which takes as arguments
 * `SpinWeighted<ComplexDataVector, N>`s for each of the Bondi arguments.
 */
template <typename IntegrandTag>
struct ComputeBondiIntegrand;

/*!
 * \brief Computes the integrand (right-hand side) of the equation which
 * determines the radial (y) dependence of the Bondi quantity \f$\beta\f$.
 *
 * \details The quantity \f$\beta\f$ is defined via the Bondi form of the
 * metric:
 * \f[
 * ds^2 = - \left(e^{2 \beta} (1 + r W) - r^2 h_{AB} U^A U^B\right) du^2 - 2
 * e^{2 \beta} du dr - 2 r^2 h_{AB} U^B du dx^A + r^2 h_{A B} dx^A dx^B. \f]
 * Additional quantities \f$J\f$ and \f$K\f$ are defined using a spherical
 * angular dyad \f$q^A\f$:
 * \f[ J \equiv h_{A B} q^A q^B, K \equiv h_{A B} q^A
 * \bar{q}^B.\f]
 * See \cite Bishop1997ik \cite Handmer2014qha for full details.
 *
 * We write the equations of motion in the compactified coordinate \f$ y \equiv
 * 1 - 2 R/ r\f$, where \f$r(u, \theta, \phi)\f$ is the Bondi radius of the
 * \f$y=\f$ constant surface and \f$R(u,\theta,\phi)\f$ is the Bondi radius of
 * the worldtube. The equation which determines \f$\beta\f$ on a surface of
 * constant \f$u\f$ given \f$J\f$ on the same surface is
 * \f[\partial_y (\beta) =
 * \frac{1}{8} (-1 + y) \left(\partial_y (J) \partial_y(\bar{J})
 * - \frac{(\partial_y (J \bar{J}))^2}{4 K^2}\right). \f]
 */
template <>
struct ComputeBondiIntegrand<Tags::Integrand<Tags::BondiBeta>> {
 public:
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiJ>, Tags::BondiJ>;
  using swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<Tags::OneMinusY>;
  using temporary_tags = tmpl::list<>;

  using return_tags = tmpl::append<tmpl::list<Tags::Integrand<Tags::BondiBeta>>,
                                   temporary_tags>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags,
                   integration_independent_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          integrand_for_beta,
      const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*integrand_for_beta)), get(args)...);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> integrand_for_beta,
      const SpinWeighted<ComplexDataVector, 2>& dy_j,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 0>& one_minus_y) noexcept;
};

/*!
 * \brief Computes the pole part of the integrand (right-hand side) of the
 * equation which determines the radial (y) dependence of the Bondi quantity
 * \f$Q\f$.
 *
 * \details The quantity \f$Q\f$ is defined via the Bondi form of the metric:
 * \f[ds^2 = - \left(e^{2 \beta} (1 + r W) - r^2 h_{AB} U^A U^B\right) du^2 - 2
 * e^{2 \beta} du dr - 2 r^2 h_{AB} U^B du dx^A + r^2 h_{A B} dx^A dx^B. \f]
 * Additional quantities \f$J\f$ and \f$K\f$ are defined using a spherical
 * angular dyad \f$q^A\f$:
 * \f[ J \equiv h_{A B} q^A q^B, K \equiv h_{A B} q^A \bar{q}^B,\f]
 * and \f$Q\f$ is defined as a supplemental variable for radial integration of
 * \f$U\f$:
 * \f[ Q_A = r^2 e^{-2\beta} h_{AB} \partial_r U^B\f]
 * and \f$Q = Q_A q^A\f$. See \cite Bishop1997ik \cite Handmer2014qha for full
 * details.
 *
 * We write the equations of motion in the compactified coordinate \f$ y \equiv
 * 1 - 2 R/ r\f$, where \f$r(u, \theta, \phi)\f$ is the Bondi radius of the
 * \f$y=\f$ constant surface and \f$R(u,\theta,\phi)\f$ is the Bondi radius of
 * the worldtube. The equation which determines \f$Q\f$ on a surface of constant
 * \f$u\f$ given \f$J\f$ and \f$\beta\f$ on the same surface is written as
 * \f[(1 - y) \partial_y Q + 2 Q = A_Q + (1 - y) B_Q.\f]
 * We refer to \f$A_Q\f$ as the "pole part" of the integrand and \f$B_Q\f$
 * as the "regular part". The pole part is computed by this function, and has
 * the expression
 * \f[A_Q = -4 \eth \beta.\f]
 */
template <>
struct ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiQ>> {
 public:
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags =
      tmpl::list<Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>>;
  using integration_independent_tags = tmpl::list<>;
  using temporary_tags = tmpl::list<>;

  using return_tags =
      tmpl::append<tmpl::list<Tags::PoleOfIntegrand<Tags::BondiQ>>,
                   temporary_tags>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags,
                   integration_independent_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          pole_of_integrand_for_q,
      const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*pole_of_integrand_for_q)), get(args)...);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
          pole_of_integrand_for_q,
      const SpinWeighted<ComplexDataVector, 1>& eth_beta) noexcept;
};

/*!
 * \brief Computes the regular part of the integrand (right-hand side) of the
 * equation which determines the radial (y) dependence of the Bondi quantity
 * \f$Q\f$.
 *
 * \details The quantity \f$Q\f$ is defined via the Bondi form of the metric:
 * \f[ds^2 = - \left(e^{2 \beta} (1 + r W) - r^2 h_{AB} U^A U^B\right) du^2 - 2
 * e^{2 \beta} du dr - 2 r^2 h_{AB} U^B du dx^A + r^2 h_{A B} dx^A dx^B. \f]
 * Additional quantities \f$J\f$ and \f$K\f$ are defined using a spherical
 * angular dyad \f$q^A\f$:
 * \f[ J \equiv h_{A B} q^A q^B, K \equiv h_{A B} q^A \bar{q}^B,\f]
 * and \f$Q\f$ is defined as a supplemental variable for radial integration of
 * \f$U\f$:
 * \f[ Q_A = r^2 e^{-2\beta} h_{AB} \partial_r U^B\f]
 * and \f$Q = Q_A q^A\f$. See \cite Bishop1997ik \cite Handmer2014qha for
 * full details.
 *
 * We write the equations of motion in the compactified coordinate \f$ y \equiv
 * 1 - 2 R/ r\f$, where \f$r(u, \theta, \phi)\f$ is the Bondi radius of the
 * \f$y=\f$ constant surface and \f$R(u,\theta,\phi)\f$ is the Bondi radius of
 * the worldtube. The equation which determines \f$Q\f$ on a surface of constant
 * \f$u\f$ given \f$J\f$ and \f$\beta\f$ on the same surface is written as
 * \f[(1 - y) \partial_y Q + 2 Q = A_Q + (1 - y) B_Q. \f]
 * We refer to \f$A_Q\f$ as the "pole part" of the integrand and \f$B_Q\f$ as
 * the "regular part". The regular part is computed by this function, and has
 * the expression
 * \f[  B_Q = - \left(2 \mathcal{A}_Q + \frac{2
 * \bar{\mathcal{A}_Q} J}{K} -  2 \partial_y (\eth (\beta)) +
 * \frac{\partial_y (\bar{\eth} (J))}{K}\right), \f]
 * where
 * \f[ \mathcal{A}_Q = - \tfrac{1}{4} \eth (\bar{J} \partial_y (J)) +
 * \tfrac{1}{4} J \partial_y (\eth (\bar{J})) -  \tfrac{1}{4} \eth (\bar{J})
 * \partial_y (J) + \frac{\eth (J \bar{J}) \partial_y (J \bar{J})}{8 K^2} -
 * \frac{\bar{J} \eth (R) \partial_y (J)}{4 R}. \f].
 */
template <>
struct ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiQ>> {
 public:
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiBeta>, Tags::Dy<Tags::BondiJ>,
                 Tags::BondiJ>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiJ>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::Ethbar>>;
  using integration_independent_tags =
      tmpl::list<Tags::EthRDividedByR, Tags::BondiK>;
  using temporary_tags =
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>;

  using return_tags =
      tmpl::append<tmpl::list<Tags::RegularIntegrand<Tags::BondiQ>>,
                   temporary_tags>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags,
                   integration_independent_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          regular_integrand_for_q,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          script_aq,
      const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*regular_integrand_for_q)),
               make_not_null(&get(*script_aq)), get(args)...);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
          regular_integrand_for_q,
      gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> script_aq,
      const SpinWeighted<ComplexDataVector, 0>& dy_beta,
      const SpinWeighted<ComplexDataVector, 2>& dy_j,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 1>& eth_dy_beta,
      const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
      const SpinWeighted<ComplexDataVector, 1>& eth_jbar_dy_j,
      const SpinWeighted<ComplexDataVector, 1>& ethbar_dy_j,
      const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
      const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
      const SpinWeighted<ComplexDataVector, 0>& k) noexcept;
};

/*!
 * \brief Computes the integrand (right-hand side) of the equation which
 * determines the radial (y) dependence of the Bondi quantity \f$U\f$.
 *
 * \details The quantity \f$U\f$ is defined via the Bondi form of the metric:
 * \f[ds^2 = - \left(e^{2 \beta} (1 + r W) - r^2 h_{AB} U^A U^B\right) du^2 - 2
 * e^{2 \beta} du dr - 2 r^2 h_{AB} U^B du dx^A + r^2 h_{A B} dx^A dx^B. \f]
 * Additional quantities \f$J\f$ and \f$K\f$ are defined using a spherical
 * angular dyad \f$q^A\f$:
 * \f[ J \equiv h_{A B} q^A q^B, K \equiv h_{A B} q^A \bar{q}^B,\f]
 * and \f$Q\f$ is defined as a supplemental variable for radial integration of
 * \f$U\f$:
 * \f[ Q_A = r^2 e^{-2\beta} h_{AB} \partial_r U^B\f]
 * and \f$U = U_A q^A\f$. See \cite Bishop1997ik \cite Handmer2014qha for full
 * details.
 *
 * We write the equations of motion in the compactified coordinate \f$ y \equiv
 * 1 - 2 R/ r\f$, where \f$r(u, \theta, \phi)\f$ is the Bondi radius of the
 * \f$y=\f$ constant surface and \f$R(u,\theta,\phi)\f$ is the Bondi radius of
 * the worldtube. The equation which determines \f$U\f$ on a surface of constant
 * \f$u\f$ given \f$J\f$, \f$\beta\f$, and \f$Q\f$ on the same surface is
 * written as
 * \f[\partial_y U = \frac{e^{2\beta}}{2 R} (K Q - J \bar{Q}). \f]
 */
template <>
struct ComputeBondiIntegrand<Tags::Integrand<Tags::BondiU>> {
 public:
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Exp2Beta, Tags::BondiJ, Tags::BondiQ>;
  using swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<Tags::BondiK, Tags::BondiR>;
  using temporary_tags = tmpl::list<>;

  using return_tags =
      tmpl::append<tmpl::list<Tags::Integrand<Tags::BondiU>>, temporary_tags>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags,
                   integration_independent_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          regular_integrand_for_u,
      const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*regular_integrand_for_u)), get(args)...);
  }

 private:
  static void apply_impl(gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
                             regular_integrand_for_u,
                         const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
                         const SpinWeighted<ComplexDataVector, 2>& j,
                         const SpinWeighted<ComplexDataVector, 1>& q,
                         const SpinWeighted<ComplexDataVector, 0>& k,
                         const SpinWeighted<ComplexDataVector, 0>& r) noexcept;
};

/*!
 * \brief Computes the pole part of the integrand (right-hand side) of the
 * equation which determines the radial (y) dependence of the Bondi quantity
 * \f$W\f$.
 *
 * \details The quantity \f$W\f$ is defined via the Bondi form of the metric:
 * \f[ds^2 = - \left(e^{2 \beta} (1 + r W) - r^2 h_{AB} U^A U^B\right) du^2 - 2
 * e^{2 \beta} du dr - 2 r^2 h_{AB} U^B du dx^A + r^2 h_{A B} dx^A dx^B. \f]
 * Additional quantities \f$J\f$ and \f$K\f$ are defined using a spherical
 * angular dyad \f$q^A\f$:
 * \f[ J \equiv h_{A B} q^A q^B, K \equiv h_{A B} q^A \bar{q}^B.\f]
 * See \cite Bishop1997ik \cite Handmer2014qha for full details.
 *
 * We write the equations of motion in the compactified coordinate \f$ y \equiv
 * 1 - 2 R/ r\f$, where \f$r(u, \theta, \phi)\f$ is the Bondi radius of the
 * \f$y=\f$ constant surface and \f$R(u,\theta,\phi)\f$ is the Bondi radius of
 * the worldtube. The equation which determines \f$W\f$ on a surface of constant
 * \f$u\f$ given \f$J\f$,\f$\beta\f$, \f$Q\f$, and \f$U\f$ on the same surface
 * is written as
 * \f[(1 - y) \partial_y W + 2 W = A_W + (1 - y) B_W.\f] We refer
 * to \f$A_W\f$ as the "pole part" of the integrand and \f$B_W\f$ as the
 * "regular part". The pole part is computed by this function, and has the
 * expression
 * \f[A_W = \eth (\bar{U}) + \bar{\eth} (U).\f]
 */
template <>
struct ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiW>> {
 public:
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<Spectral::Swsh::Tags::Derivative<
      Tags::BondiU, Spectral::Swsh::Tags::Ethbar>>;
  using integration_independent_tags = tmpl::list<>;
  using temporary_tags = tmpl::list<>;

  using return_tags =
      tmpl::append<tmpl::list<Tags::PoleOfIntegrand<Tags::BondiW>>,
                   temporary_tags>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags,
                   integration_independent_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          pole_of_integrand_for_w,
      const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*pole_of_integrand_for_w)), get(args)...);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
          pole_of_integrand_for_w,
      const SpinWeighted<ComplexDataVector, 0>& ethbar_u) noexcept;
};

/*!
 * \brief Computes the regular part of the integrand (right-hand side) of the
 * equation which determines the radial (y) dependence of the Bondi quantity
 * \f$W\f$.
 *
 * \details The quantity \f$W\f$ is defined via the Bondi form of the metric:
 * \f[ds^2 = - \left(e^{2 \beta} (1 + r W) - r^2 h_{AB} U^A U^B\right) du^2 - 2
 * e^{2 \beta} du dr - 2 r^2 h_{AB} U^B du dx^A + r^2 h_{A B} dx^A dx^B. \f]
 * Additional quantities \f$J\f$ and \f$K\f$ are defined using a spherical
 * angular dyad \f$q^A\f$:
 * \f[ J \equiv h_{A B} q^A q^B, K \equiv h_{A B} q^A \bar{q}^B,\f]
 * See \cite Bishop1997ik \cite Handmer2014qha for full details.
 *
 * We write the equations of motion in the compactified coordinate \f$ y \equiv
 * 1 - 2 R/ r\f$, where \f$r(u, \theta, \phi)\f$ is the Bondi radius of the
 * \f$y=\f$ constant surface and \f$R(u,\theta,\phi)\f$ is the Bondi radius of
 * the worldtube. The equation which determines \f$W\f$ on a surface of constant
 * \f$u\f$ given \f$J\f$, \f$\beta\f$, \f$Q\f$, \f$U\f$ on the same surface is
 * written as
 * \f[(1 - y) \partial_y W + 2 W = A_W + (1 - y) B_W. \f]
 * We refer to \f$A_W\f$ as the "pole part" of the integrand and \f$B_W\f$ as
 * the "regular part". The regular part is computed by this function, and has
 * the expression
 * \f[  B_W = \tfrac{1}{4} \partial_y (\eth (\bar{U})) + \tfrac{1}{4} \partial_y
 * (\bar{\eth} (U)) -  \frac{1}{2 R} + \frac{e^{2 \beta} (\mathcal{A}_W +
 * \bar{\mathcal{A}_W})}{4 R}, \f]
 * where
 * \f{align*}
 * \mathcal{A}_W =& - \eth (\beta) \eth (\bar{J}) + \tfrac{1}{2} \bar{\eth}
 * (\bar{\eth} (J)) + 2 \bar{\eth} (\beta) \bar{\eth} (J) + (\bar{\eth}
 * (\beta))^2 J + \bar{\eth}
 * (\bar{\eth} (\beta)) J + \frac{\eth (J \bar{J}) \bar{\eth} (J \bar{J})}{8
 * K^3} + \frac{1}{2 K} -  \frac{\eth (\bar{\eth} (J \bar{J}))}{8 K} -
 * \frac{\eth (J
 * \bar{J}) \bar{\eth} (\beta)}{2 K} \nonumber \\
 * &-  \frac{\eth (\bar{J}) \bar{\eth} (J)}{4 K} -  \frac{\eth (\bar{\eth} (J))
 * \bar{J}}{4 K} + \tfrac{1}{2} K  -  \eth (\bar{\eth} (\beta)) K -  \eth
 * (\beta) \bar{\eth} (\beta) K + \tfrac{1}{4} (- K Q \bar{Q} + J \bar{Q}^2).
 * \f}
 */
template <>
struct ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiW>> {
 public:
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiU>, Tags::Exp2Beta, Tags::BondiJ,
                 Tags::BondiQ>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::EthEth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::EthEthbar>,
      Spectral::Swsh::Tags::Derivative<
          Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                           Spectral::Swsh::Tags::Ethbar>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::EthEthbar>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiU>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::EthbarEthbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::Ethbar>>;
  using integration_independent_tags =
      tmpl::list<Tags::EthRDividedByR, Tags::BondiK, Tags::BondiR>;
  using temporary_tags =
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>>;

  using return_tags =
      tmpl::append<tmpl::list<Tags::RegularIntegrand<Tags::BondiW>>,
                   temporary_tags>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags,
                   integration_independent_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          regular_integrand_for_w,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          script_av,
      const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*regular_integrand_for_w)),
               make_not_null(&get(*script_av)), get(args)...);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
          regular_integrand_for_w,
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> script_av,
      const SpinWeighted<ComplexDataVector, 1>& dy_u,
      const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 1>& q,
      const SpinWeighted<ComplexDataVector, 1>& eth_beta,
      const SpinWeighted<ComplexDataVector, 2>& eth_eth_beta,
      const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_beta,
      const SpinWeighted<ComplexDataVector, 2>& eth_ethbar_j,
      const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_j_jbar,
      const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
      const SpinWeighted<ComplexDataVector, 0>& ethbar_dy_u,
      const SpinWeighted<ComplexDataVector, 0>& ethbar_ethbar_j,
      const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
      const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
      const SpinWeighted<ComplexDataVector, 0>& k,
      const SpinWeighted<ComplexDataVector, 0>& r) noexcept;
};

/*!
 * \brief Computes the pole part of the integrand (right-hand side) of the
 * equation which determines the radial (y) dependence of the Bondi quantity
 * \f$H\f$.
 *
 * \details The quantity \f$H \equiv \partial_u J\f$ (evaluated at constant y)
 * is defined via the Bondi form of the metric:
 * \f[ds^2 = - \left(e^{2 \beta} (1 + r W) - r^2 h_{AB} U^A U^B\right) du^2 - 2
 * e^{2 \beta} du dr - 2 r^2 h_{AB} U^B du dx^A + r^2 h_{A B} dx^A dx^B. \f]
 * Additional quantities \f$J\f$ and \f$K\f$ are defined using a spherical
 * angular dyad \f$q^A\f$:
 * \f[ J \equiv h_{A B} q^A q^B, K \equiv h_{A B} q^A \bar{q}^B.\f]
 * See \cite Bishop1997ik \cite Handmer2014qha for full details.
 *
 * We write the equations of motion in the compactified coordinate \f$ y \equiv
 * 1 - 2 R/ r\f$, where \f$r(u, \theta, \phi)\f$ is the Bondi radius of the
 * \f$y=\f$ constant surface and \f$R(u,\theta,\phi)\f$ is the Bondi radius of
 * the worldtube. The equation which determines \f$W\f$ on a surface of constant
 * \f$u\f$ given \f$J\f$,\f$\beta\f$, \f$Q\f$, \f$U\f$, and \f$W\f$ on the same
 * surface is written as
 * \f[(1 - y) \partial_y H + H + (1 - y)(\mathcal{D}_J H
 * + \bar{\mathcal{D}}_J \bar{H}) = A_J + (1 - y) B_J.\f]
 *
 * We refer to \f$A_J\f$ as the "pole part" of the integrand
 * and \f$B_J\f$ as the "regular part". The pole part is computed by this
 * function, and has the expression
 * \f{align*}
 * A_J =& - \tfrac{1}{2} \eth (J \bar{U}) -  \eth (\bar{U}) J -  \tfrac{1}{2}
 * \bar{\eth} (U) J -  \eth (U) K -  \tfrac{1}{2} (\bar{\eth} (J) U) + 2 J W
 * \f}
 */
template <>
struct ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiH>> {
 public:
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::BondiJ, Tags::BondiU, Tags::BondiW>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::BondiU, Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::BondiU>,
          Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                       Spectral::Swsh::Tags::Ethbar>>;
  using integration_independent_tags = tmpl::list<Tags::BondiK>;
  using temporary_tags = tmpl::list<>;

  using return_tags =
      tmpl::append<tmpl::list<Tags::PoleOfIntegrand<Tags::BondiH>>,
                   temporary_tags>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags,
                   integration_independent_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          pole_of_integrand_for_h,
      const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*pole_of_integrand_for_h)), get(args)...);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 2>*>
          pole_of_integrand_for_h,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 1>& u,
      const SpinWeighted<ComplexDataVector, 0>& w,
      const SpinWeighted<ComplexDataVector, 2>& eth_u,
      const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
      const SpinWeighted<ComplexDataVector, -2>& ethbar_jbar_u,
      const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
      const SpinWeighted<ComplexDataVector, 0>& k) noexcept;
};

/*!
 * \brief Computes the pole part of the integrand (right-hand side) of the
 * equation which determines the radial (y) dependence of the Bondi quantity
 * \f$H\f$.
 *
 * \details The quantity \f$H \equiv \partial_u J\f$ (evaluated at constant y)
 * is defined via the Bondi form of the metric:
 * \f[ds^2 = - \left(e^{2 \beta} (1 + r W) - r^2 h_{AB} U^A U^B\right) du^2 - 2
 * e^{2 \beta} du dr - 2 r^2 h_{AB} U^B du dx^A + r^2 h_{A B} dx^A dx^B. \f]
 * Additional quantities \f$J\f$ and \f$K\f$ are defined using a spherical
 * angular dyad \f$q^A\f$:
 * \f[ J \equiv h_{A B} q^A q^B, K \equiv h_{A B} q^A \bar{q}^B.\f]
 * See \cite Bishop1997ik \cite Handmer2014qha for full details.
 *
 * We write the equations of motion in the compactified coordinate \f$ y \equiv
 * 1 - 2 R/ r\f$, where \f$r(u, \theta, \phi)\f$ is the Bondi radius of the
 * \f$y=\f$ constant surface and \f$R(u,\theta,\phi)\f$ is the Bondi radius of
 * the worldtube. The equation which determines \f$H\f$ on a surface of constant
 * \f$u\f$ given \f$J\f$,\f$\beta\f$, \f$Q\f$, \f$U\f$, and \f$W\f$ on the same
 * surface is written as
 * \f[(1 - y) \partial_y H + H + (1 - y)(\mathcal{D}_J H + \bar{\mathcal{D}}_J
 * \bar{H}) = A_J + (1 - y) B_J.\f]
 * We refer to \f$A_J\f$ as the "pole part" of the integrand
 * and \f$B_J\f$ as the "regular part". The pole part is computed by this
 * function, and has the expression
 * \f{align*}
 * B_J =& -\tfrac{1}{2} \left(\eth(\partial_y (J) \bar{U}) +  \partial_y
 * (\bar{\eth} (J)) U \right) + J (\mathcal{B}_J + \bar{\mathcal{B}}_J) \notag\\
 * &+ \frac{e^{2 \beta}}{2 R} \left(\mathcal{C}_J + \eth (\eth (\beta)) -
 * \tfrac{1}{2} \eth (Q) -  (\mathcal{A}_J + \bar{\mathcal{A}_J}) J +
 * \frac{\bar{\mathcal{C}_J} J^2}{K^2} + \frac{\eth (J (-2 \bar{\eth} (\beta) +
 * \bar{Q}))}{4 K} -  \frac{\eth (\bar{Q}) J}{4 K} + (\eth (\beta) -
 * \tfrac{1}{2} Q)^2\right) \notag\\
 * &-  \partial_y (J)  \left(\frac{\eth (U) \bar{J}}{2 K} -  \tfrac{1}{2}
 * \bar{\eth} (\bar{U}) J K + \tfrac{1}{4} (\eth (\bar{U}) -  \bar{\eth} (U))
 * K^2 + \frac{1}{2} \frac{\eth (R) \bar{U}}{R} -  \frac{1}{2}
 * W\right)\notag\\
 * &+  \partial_y (\bar{J}) \left(- \tfrac{1}{4} (- \eth (\bar{U}) + \bar{\eth}
 * (U)) J^2 + \eth (U) J \left(- \frac{1}{2 K} + \tfrac{1}{2}
 * K\right)\right)\notag\\
 * &+ (1 - y) \bigg[\frac{1}{2} \left(- \frac{\partial_y (J)}{R} + \frac{2
 * \partial_{u} (R) \partial_y (\partial_y (J))}{R} + \partial_y
 * (\partial_y (J)) W\right)  + \partial_y (J) \left(\tfrac{1}{2} \partial_y (W)
 * + \frac{1}{2 R}\right)\bigg]\notag\\
 * &+ (1 - y)^2 \bigg[ \frac{\partial_y (\partial_y (J)) }{4 R} \bigg],
 * \f}
 * where
 * \f{align*}
 * \mathcal{A}_J =& \tfrac{1}{4} \eth (\eth (\bar{J})) -  \frac{1}{4 K^3} -
 * \frac{\eth (\bar{\eth} (J \bar{J})) -  (\eth (\bar{\eth} (\bar{J})) - 4
 \bar{J})
 * J}{16 K^3} + \frac{3}{4 K} -  \frac{\eth (\bar{\eth} (\beta))}{4 K} \notag\\
 * &-  \frac{\eth (\bar{\eth} (J)) \bar{J} (1 -  \frac{1}{4 K^2})}{4 K} +
 * \tfrac{1}{2} \eth (\bar{J}) \left(\eth (\beta) + \frac{\bar{\eth} (J \bar{J})
 * J}{4 K^3} -  \frac{\bar{\eth} (J) (-1 + 2 K^2)}{4 K^3} -  \tfrac{1}{2}
 * Q\right)\\
 * \mathcal{B}_J =& - \frac{\eth (U) \bar{J} \partial_y (J \bar{J})}{4 K} +
 * \tfrac{1}{2} \partial_y (W) + \frac{1}{4 R} + \tfrac{1}{4} \bar{\eth} (J)
 * \partial_y (\bar{J}) U -  \frac{\bar{\eth} (J \bar{J}) \partial_y (J \bar{J})
 * U}{8 K^2} \notag\\&-  \tfrac{1}{4} J \partial_y (\eth (\bar{J})) \bar{U} +
 * \tfrac{1}{4} (\eth (J \partial_y (\bar{J})) + \frac{J \eth (R)
 * \partial_y(\bar{J})}{R}) \bar{U} \\
 * &+ (1 - y) \bigg[ \frac{\mathcal{D}_J \partial_{u} (R)
 * \partial_y (J)}{R} -  \tfrac{1}{4} \partial_y (J) \partial_y (\bar{J}) W +
 * \frac{(\partial_y (J \bar{J}))^2 W}{16 K^2} \bigg] \\
 * &+ (1 - y)^2 \bigg[ - \frac{\partial_y (J) \partial_y (\bar{J})}{8 R} +
 * \frac{(\partial_y (J \bar{J}))^2}{32 K^2 R} \bigg]\\
 * \mathcal{C}_J =& \tfrac{1}{2} \bar{\eth} (J) K (\eth (\beta) -  \tfrac{1}{2}
 * Q)\\ \mathcal{D}_J =& \tfrac{1}{4} \left(-2 \partial_y (\bar{J}) +
 * \frac{\bar{J} \partial_y (J \bar{J})}{K^2}\right)
 * \f}
 */
template <>
struct ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiH>> {
 public:
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::Dy<Tags::BondiJ>>, Tags::Dy<Tags::BondiJ>,
                 Tags::Dy<Tags::BondiW>, Tags::Exp2Beta, Tags::BondiJ,
                 Tags::BondiQ, Tags::BondiU, Tags::BondiW>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::EthEth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::EthEthbar>,
      Spectral::Swsh::Tags::Derivative<
          Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                           Spectral::Swsh::Tags::Ethbar>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::EthEthbar>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiQ, Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU, Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiUbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiJ>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::EthbarEthbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::JbarQMinus2EthBeta,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiQ,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                       Spectral::Swsh::Tags::Ethbar>>;
  using integration_independent_tags =
      tmpl::list<Tags::DuRDividedByR, Tags::EthRDividedByR, Tags::BondiK,
                 Tags::OneMinusY, Tags::BondiR>;
  using temporary_tags =
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 2>>>;

  using return_tags =
      tmpl::append<tmpl::list<Tags::RegularIntegrand<Tags::BondiH>>,
                   temporary_tags>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags,
                   integration_independent_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          regular_integrand_for_h,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          script_aj,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          script_bj,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          script_cj,
      const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*regular_integrand_for_h)),
               make_not_null(&get(*script_aj)), make_not_null(&get(*script_bj)),
               make_not_null(&get(*script_cj)), get(args)...);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 2>*>
          regular_integrand_for_h,
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> script_aj,
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> script_bj,
      gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> script_cj,
      const SpinWeighted<ComplexDataVector, 2>& dy_dy_j,
      const SpinWeighted<ComplexDataVector, 2>& dy_j,
      const SpinWeighted<ComplexDataVector, 0>& dy_w,
      const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 1>& q,
      const SpinWeighted<ComplexDataVector, 1>& u,
      const SpinWeighted<ComplexDataVector, 0>& w,
      const SpinWeighted<ComplexDataVector, 1>& eth_beta,
      const SpinWeighted<ComplexDataVector, 2>& eth_eth_beta,
      const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_beta,
      const SpinWeighted<ComplexDataVector, 2>& eth_ethbar_j,
      const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_j_jbar,
      const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
      const SpinWeighted<ComplexDataVector, 2>& eth_q,
      const SpinWeighted<ComplexDataVector, 2>& eth_u,
      const SpinWeighted<ComplexDataVector, 2>& eth_ubar_dy_j,
      const SpinWeighted<ComplexDataVector, 1>& ethbar_dy_j,
      const SpinWeighted<ComplexDataVector, 0>& ethbar_ethbar_j,
      const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
      const SpinWeighted<ComplexDataVector, -1>& ethbar_jbar_dy_j,
      const SpinWeighted<ComplexDataVector, -2>& ethbar_jbar_q_minus_2_eth_beta,
      const SpinWeighted<ComplexDataVector, 0>& ethbar_q,
      const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
      const SpinWeighted<ComplexDataVector, 0>& du_r_divided_by_r,
      const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
      const SpinWeighted<ComplexDataVector, 0>& k,
      const SpinWeighted<ComplexDataVector, 0>& one_minus_y,
      const SpinWeighted<ComplexDataVector, 0>& r) noexcept;
};

/*!
 * \brief Computes the linear factor which multiplies \f$H\f$ in the
 * equation which determines the radial (y) dependence of the Bondi quantity
 * \f$H\f$.
 *
 * \details The quantity \f$H \equiv \partial_u J\f$ (evaluated at constant y)
 * is defined via the Bondi form of the metric:
 * \f[ds^2 = - \left(e^{2 \beta} (1 + r W) - r^2 h_{AB} U^A U^B\right) du^2 - 2
 * e^{2 \beta} du dr - 2 r^2 h_{AB} U^B du dx^A + r^2 h_{A B} dx^A dx^B. \f]
 * Additional quantities \f$J\f$ and \f$K\f$ are defined using a spherical
 * angular dyad \f$q^A\f$:
 * \f[ J \equiv h_{A B} q^A q^B, K \equiv h_{A B} q^A \bar{q}^B.\f]
 * See \cite Bishop1997ik \cite Handmer2014qha for full details.
 *
 * We write the equations of motion in the compactified coordinate \f$ y \equiv
 * 1 - 2 R/ r\f$, where \f$r(u, \theta, \phi)\f$ is the Bondi radius of the
 * \f$y=\f$ constant surface and \f$R(u,\theta,\phi)\f$ is the Bondi radius of
 * the worldtube. The equation which determines \f$H\f$ on a surface of constant
 * \f$u\f$ given \f$J\f$,\f$\beta\f$, \f$Q\f$, \f$U\f$, and \f$W\f$ on the same
 * surface is written as
 * \f[(1 - y) \partial_y H + H + (1 - y) J (\mathcal{D}_J
 * H + \bar{\mathcal{D}}_J \bar{H}) = A_J + (1 - y) B_J.\f]
 * The quantity \f$1 +(1 - y) J \mathcal{D}_J\f$ is the linear factor
 * for the non-conjugated \f$H\f$, and is computed from the equation:
 * \f[\mathcal{D}_J = \frac{1}{4}(-2 \partial_y \bar{J} + \frac{\bar{J}
 * \partial_y (J \bar{J})}{K^2})\f]
 */
template <>
struct ComputeBondiIntegrand<Tags::LinearFactor<Tags::BondiH>> {
 public:
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiJ>, Tags::BondiJ>;
  using swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<Tags::OneMinusY>;
  using temporary_tags =
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 2>>>;

  using return_tags = tmpl::append<tmpl::list<Tags::LinearFactor<Tags::BondiH>>,
                                   temporary_tags>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags,
                   integration_independent_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          linear_factor_for_h,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          script_djbar,
      const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*linear_factor_for_h)),
               make_not_null(&get(*script_djbar)), get(args)...);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> linear_factor_for_h,
      gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> script_djbar,
      const SpinWeighted<ComplexDataVector, 2>& dy_j,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 0>& one_minus_y) noexcept;
};

/*!
 * \brief Computes the linear factor which multiplies \f$\bar{H}\f$ in the
 * equation which determines the radial (y) dependence of the Bondi quantity
 * \f$H\f$.
 *
 * \details The quantity \f$H \equiv \partial_u J\f$ (evaluated at constant y)
 * is defined via the Bondi form of the metric:
 * \f[ds^2 = - \left(e^{2 \beta} (1 + r W) - r^2 h_{AB} U^A U^B\right) du^2 - 2
 * e^{2 \beta} du dr - 2 r^2 h_{AB} U^B du dx^A + r^2 h_{A B} dx^A dx^B. \f]
 * Additional quantities \f$J\f$ and \f$K\f$ are defined using a spherical
 * angular dyad \f$q^A\f$:
 * \f[ J \equiv h_{A B} q^A q^B, K \equiv h_{A B} q^A \bar{q}^B.\f]
 * See \cite Bishop1997ik \cite Handmer2014qha for full details.
 *
 * We write the equations of motion in the compactified coordinate \f$ y \equiv
 * 1 - 2 R/ r\f$, where \f$r(u, \theta, \phi)\f$ is the Bondi radius of the
 * \f$y=\f$ constant surface and \f$R(u,\theta,\phi)\f$ is the Bondi radius of
 * the worldtube. The equation which determines \f$H\f$ on a surface of constant
 * \f$u\f$ given \f$J\f$,\f$\beta\f$, \f$Q\f$, \f$U\f$, and \f$W\f$ on the same
 * surface is written as
 * \f[(1 - y) \partial_y H + H + (1 - y) J (\mathcal{D}_J H +
 * \bar{\mathcal{D}}_J \bar{H}) = A_J + (1 - y) B_J.\f]
 * The quantity \f$ (1 - y) J \bar{\mathcal{D}}_J\f$ is the linear factor
 * for the non-conjugated \f$H\f$, and is computed from the equation:
 * \f[\mathcal{D}_J = \frac{1}{4}(-2 \partial_y \bar{J} + \frac{\bar{J}
 * \partial_y (J \bar{J})}{K^2})\f]
 */
template <>
struct ComputeBondiIntegrand<Tags::LinearFactorForConjugate<Tags::BondiH>> {
 public:
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiJ>, Tags::BondiJ>;
  using swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<Tags::OneMinusY>;
  using temporary_tags =
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 2>>>;

  using return_tags =
      tmpl::append<tmpl::list<Tags::LinearFactorForConjugate<Tags::BondiH>>,
                   temporary_tags>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags,
                   integration_independent_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 4>>*>
          linear_factor_for_conjugate_h,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          script_djbar,
      const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*linear_factor_for_conjugate_h)),
               make_not_null(&get(*script_djbar)), get(args)...);
  }

 private:
  static void apply_impl(
      gsl::not_null<SpinWeighted<ComplexDataVector, 4>*>
          linear_factor_for_conjugate_h,
      gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> script_djbar,
      const SpinWeighted<ComplexDataVector, 2>& dy_j,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 0>& one_minus_y) noexcept;
};
}  // namespace Cce
