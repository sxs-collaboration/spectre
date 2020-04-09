// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace Cce {
namespace detail {
// Precomputation routines for supplying the additional quantities necessary
// to correct the output of the angular derivative routines from the angular
// derivatives evaluated at constant numerical coordinates (which is what is
// returned after the libsharp evaluation) to the angular derivatives at
// constant Bondi radius (which is what appears in the literature equations
// and is simple to combine to obtain the hypersurface integrands).
//
// Warning: this 'on demand' template is a way of taking advantage of the blaze
// expression templates in a generic, modular way. However, this can be
// dangerous. The returned value MUST be a blaze expression template directly,
// and not a wrapper type (like `SpinWeighted`). Otherwise, some information is
// lost on the stack and the expression template is corrupted. So, in these 'on
// demand' returns, the arguments must be fully unpacked to vector types.
//
// The `Select` operand is a `std::bool_constant` that ensures mutual
// exclusivity of the template specializations.
template <typename Tag, typename SpinConstant, typename Select>
struct OnDemandInputsForSwshJacobianImpl;

// default to just retrieving it from the box if not providing an expression
// template shortcut
template <typename Tag>
struct OnDemandInputsForSwshJacobianImpl<
    Tags::Dy<Tag>, std::integral_constant<int, Tag::type::type::spin>,
    std::bool_constant<not tt::is_a_v<::Tags::Multiplies, Tag> and
                       not tt::is_a_v<Tags::Dy, Tag>>> {
  template <typename DataBoxTagList>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(
      const db::DataBox<DataBoxTagList>& box) noexcept {
    return get(db::get<Tags::Dy<Tag>>(box)).data();
  }
};

// default to retrieving from the box if the requested tag is a second
// derivative without an additional evaluation channel
template <typename Tag>
struct OnDemandInputsForSwshJacobianImpl<
    Tags::Dy<Tags::Dy<Tag>>, std::integral_constant<int, Tag::type::type::spin>,
    std::bool_constant<not tt::is_a_v<::Tags::Multiplies, Tag>>> {
  template <typename DataBoxTagList>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(
      const db::DataBox<DataBoxTagList>& box) noexcept {
    return get(db::get<Tags::Dy<Tags::Dy<Tag>>>(box)).data();
  }
};

// use the product rule to provide an expression template for derivatives of
// products
template <typename LhsTag, typename RhsTag>
struct OnDemandInputsForSwshJacobianImpl<
    Tags::Dy<::Tags::Multiplies<LhsTag, RhsTag>>,
    std::integral_constant<int,
                           LhsTag::type::type::spin + RhsTag::type::type::spin>,
    std::bool_constant<not std::is_same_v<LhsTag, Tags::BondiJbar> and
                       not std::is_same_v<LhsTag, Tags::BondiUbar> and
                       not std::is_same_v<RhsTag, Tags::BondiJbar>>> {
  template <typename DataBoxTagList>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(
      const db::DataBox<DataBoxTagList>& box) noexcept {
    decltype(auto) lhs = get(db::get<LhsTag>(box)).data();
    decltype(auto) dy_lhs = get(db::get<Tags::Dy<LhsTag>>(box)).data();
    decltype(auto) rhs = get(db::get<RhsTag>(box)).data();
    decltype(auto) dy_rhs = get(db::get<Tags::Dy<RhsTag>>(box)).data();
    return lhs * dy_rhs + dy_lhs * rhs;
  }
};

// use the product rule and an explicit conjugate to provide an expression
// template for derivatives of products with `Tags::BondiJbar` as the right-hand
// operand
template <typename LhsTag>
struct OnDemandInputsForSwshJacobianImpl<
    Tags::Dy<::Tags::Multiplies<LhsTag, Tags::BondiJbar>>,
    std::integral_constant<int, LhsTag::type::type::spin - 2>, std::true_type> {
  template <typename DataBoxTagList>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(
      const db::DataBox<DataBoxTagList>& box) noexcept {
    decltype(auto) lhs = get(get<LhsTag>(box)).data();
    decltype(auto) dy_lhs = get(get<Tags::Dy<LhsTag>>(box)).data();
    decltype(auto) jbar = conj(get(get<Tags::BondiJ>(box)).data());
    decltype(auto) dy_jbar = conj(get(get<Tags::Dy<Tags::BondiJ>>(box)).data());
    return lhs * dy_jbar + dy_lhs * jbar;
  }
};

// use the product rule and an explicit conjugate to provide an expression
// template for derivatives of products with `Tags::BondiJbar` as the left-hand
// operand.
template <typename RhsTag>
struct OnDemandInputsForSwshJacobianImpl<
    Tags::Dy<::Tags::Multiplies<Tags::BondiJbar, RhsTag>>,
    std::integral_constant<int, RhsTag::type::type::spin - 2>, std::true_type> {
  template <typename DataBoxTagList>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(
      const db::DataBox<DataBoxTagList>& box) noexcept {
    decltype(auto) rhs = get(get<RhsTag>(box)).data();
    decltype(auto) dy_rhs = get(get<Tags::Dy<RhsTag>>(box)).data();
    decltype(auto) jbar = conj(get(get<Tags::BondiJ>(box)).data());
    decltype(auto) dy_jbar = conj(get(get<Tags::Dy<Tags::BondiJ>>(box)).data());
    return dy_jbar * rhs + jbar * dy_rhs;
  }
};

// use the product rule and an explicit conjugate to provide an expression
// template for derivatives of products with `Tags::BondiUbar` as the left-hand
// operand.
template <typename RhsTag>
struct OnDemandInputsForSwshJacobianImpl<
    Tags::Dy<::Tags::Multiplies<Tags::BondiUbar, RhsTag>>,
    std::integral_constant<int, RhsTag::type::type::spin - 1>, std::true_type> {
  template <typename DataBoxTagList>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(
      const db::DataBox<DataBoxTagList>& box) noexcept {
    decltype(auto) ubar = conj(get(get<Tags::BondiU>(box)).data());
    decltype(auto) dy_ubar = conj(get(get<Tags::Dy<Tags::BondiU>>(box)).data());
    decltype(auto) rhs = get(get<RhsTag>(box)).data();
    decltype(auto) dy_rhs = get(get<Tags::Dy<RhsTag>>(box)).data();
    return ubar * dy_rhs + dy_ubar * rhs;
  }
};

// use the product rule and an explicit conjugate to provide an expression
// template for second derivatives of products with `Tags::BondiJbar` as the
// right-hand operand.
template <typename LhsTag>
struct OnDemandInputsForSwshJacobianImpl<
    Tags::Dy<Tags::Dy<::Tags::Multiplies<LhsTag, Tags::BondiJbar>>>,
    std::integral_constant<int, LhsTag::type::type::spin - 2>, std::true_type> {
  template <typename DataBoxTagList>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(
      const db::DataBox<DataBoxTagList>& box) noexcept {
    decltype(auto) lhs = get(get<LhsTag>(box)).data();
    decltype(auto) dy_lhs = get(get<Tags::Dy<LhsTag>>(box)).data();
    decltype(auto) dy_dy_lhs = get(get<Tags::Dy<Tags::Dy<LhsTag>>>(box)).data();
    decltype(auto) jbar = conj(get(get<Tags::BondiJ>(box)).data());
    decltype(auto) dy_jbar = conj(get(get<Tags::Dy<Tags::BondiJ>>(box)).data());
    decltype(auto) dy_dy_jbar =
        conj(get(get<Tags::Dy<Tags::Dy<Tags::BondiJ>>>(box)).data());
    return lhs * dy_dy_jbar + 2.0 * dy_lhs * dy_jbar + dy_dy_lhs * jbar;
  }
};

// default to extracting directly from the box for spin-weighted derivatives of
// radial (y) derivatives
template <typename Tag, typename DerivKind>
struct OnDemandInputsForSwshJacobianImpl<
    Spectral::Swsh::Tags::Derivative<Tags::Dy<Tag>, DerivKind>,
    std::integral_constant<
        int, Spectral::Swsh::Tags::Derivative<Tags::Dy<Tag>, DerivKind>::spin>,
    std::true_type> {
  template <typename DataBoxTagList>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(
      const db::DataBox<DataBoxTagList>& box) noexcept {
    return get(get<Spectral::Swsh::Tags::Derivative<Tags::Dy<Tag>, DerivKind>>(
                   box))
        .data();
  }
};

// compute the derivative of the `jbar * (q - 2 eth_beta)` using the product
// rule and the commutation rule for the partials with respect to y and the
// spin_weighted derivatives
template <>
struct OnDemandInputsForSwshJacobianImpl<Tags::Dy<Tags::JbarQMinus2EthBeta>,
                                         std::integral_constant<int, -1>,
                                         std::true_type> {
  template <typename DataBoxTagList>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(
      const db::DataBox<DataBoxTagList>& box) noexcept {
    decltype(auto) dy_beta = get(get<Tags::Dy<Tags::BondiBeta>>(box)).data();
    decltype(auto) dy_j = get(get<Tags::Dy<Tags::BondiJ>>(box)).data();
    decltype(auto) dy_q = get(get<Tags::Dy<Tags::BondiQ>>(box)).data();
    decltype(auto) eth_beta =
        get(get<Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                 Spectral::Swsh::Tags::Eth>>(
                box))
            .data();
    decltype(auto) eth_dy_beta =
        get(get<Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                                 Spectral::Swsh::Tags::Eth>>(
                box))
            .data();
    decltype(auto) eth_r_divided_by_r =
        get(get<Tags::EthRDividedByR>(box)).data();
    decltype(auto) j = get(get<Tags::BondiJ>(box)).data();
    decltype(auto) q = get(get<Tags::BondiQ>(box)).data();
    return conj(j) * dy_q + conj(dy_j) * q - 2.0 * conj(j) * eth_dy_beta -
           2.0 * eth_beta * conj(dy_j) -
           2.0 * conj(j) * eth_r_divided_by_r * dy_beta;
  }
};
}  // namespace detail

/// Provide an expression template or reference to `Tag`, intended for
/// situations for which a repeated computation is more
/// desirable than storing a value in the \ref DataBoxGroup (e.g. for
/// conjugation and simple product rule expansion).
template <typename Tag>
using OnDemandInputsForSwshJacobian = detail::OnDemandInputsForSwshJacobianImpl<
    Tag, std::integral_constant<int, Tag::type::type::spin>, std::true_type>;

/*!
 * \brief Performs a mutation to a spin-weighted spherical harmonic derivative
 * value from the numerical coordinate (the spin-weighted derivative at
 * fixed \f$y\f$) to the Bondi coordinates (the spin-weighted derivative at
 * fixed \f$r\f$), inplace to the requested tag.
 *
 * \details This should be performed only once for each derivative evaluation
 * for each tag, as a repeated inplace evaluation will compound and result in
 * incorrect values in the \ref DataBoxGroup. This is compatible with acting as
 * a mutation in `db::mutate_apply`.
 * \note In each specialization, there is an additional type alias
 * `on_demand_argument_tags` that contains tags that represent additional
 * quantities to be passed as arguments that need not be in the \ref
 * DataBoxGroup. These quantities are suggested to be evaluated by the 'on
 * demand' mechanism provided by `Cce::OnDemandInputsForSwshJacobian`, which
 * provides the additional quantities as blaze expression templates rather than
 * unnecessarily caching intermediate results that aren't re-used.
 */
template <typename DerivativeTag>
struct ApplySwshJacobianInplace;

/*!
 * \brief Specialization for the spin-weighted derivative \f$\eth\f$.
 *
 * \details The implemented equation is:
 *
 * \f[ \eth F = \eth^\prime F - (1 - y) \frac{\eth R}{R} \partial_y F,
 * \f]
 *
 * where \f$\eth\f$ is the derivative at constant Bondi radius \f$r\f$ and
 * \f$\eth^\prime\f$ is the derivative at constant numerical radius \f$y\f$.
 */
template <typename ArgumentTag>
struct ApplySwshJacobianInplace<
    Spectral::Swsh::Tags::Derivative<ArgumentTag, Spectral::Swsh::Tags::Eth>> {
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags =
      tmpl::list<Tags::OneMinusY, Tags::EthRDividedByR>;

  using return_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<ArgumentTag, Spectral::Swsh::Tags::Eth>>;
  using argument_tags = tmpl::append<integration_independent_tags>;
  using on_demand_argument_tags = tmpl::list<Tags::Dy<ArgumentTag>>;

  static constexpr int spin =
      Spectral::Swsh::Tags::Derivative<ArgumentTag,
                                       Spectral::Swsh::Tags::Eth>::spin;
  template <typename DyArgumentType>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, spin>>*>
          eth_argument,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const DyArgumentType& dy_argument) {
    get(*eth_argument) -= get(one_minus_y) * get(eth_r_divided_by_r) *
                          SpinWeighted<DyArgumentType, spin - 1>{dy_argument};
  }
};

/*!
 * \brief Specialization for the spin-weighted derivative \f$\bar{\eth}\f$.
 *
 * \details The implemented equation is:
 *
 * \f[
 * \bar{\eth} F = \bar{\eth}^\prime F
 * - (1 - y) \frac{\bar{\eth} R}{R} \partial_y F,
 *\f]
 *
 * where \f$\bar{\eth}\f$ is the derivative at constant Bondi radius \f$r\f$ and
 * \f$\bar{\eth}^\prime\f$ is the derivative at constant numerical radius
 * \f$y\f$.
 */
template <typename ArgumentTag>
struct ApplySwshJacobianInplace<Spectral::Swsh::Tags::Derivative<
    ArgumentTag, Spectral::Swsh::Tags::Ethbar>> {
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags =
      tmpl::list<Tags::OneMinusY, Tags::EthRDividedByR>;

  using return_tags = tmpl::list<Spectral::Swsh::Tags::Derivative<
      ArgumentTag, Spectral::Swsh::Tags::Ethbar>>;
  using argument_tags = tmpl::append<integration_independent_tags>;
  using on_demand_argument_tags = tmpl::list<Tags::Dy<ArgumentTag>>;

  static constexpr int spin =
      Spectral::Swsh::Tags::Derivative<ArgumentTag,
                                       Spectral::Swsh::Tags::Ethbar>::spin;
  template <typename DyArgumentType>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, spin>>*>
          ethbar_argument,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const DyArgumentType& dy_argument) {
    get(*ethbar_argument) -=
        get(one_minus_y) * conj(get(eth_r_divided_by_r)) *
        SpinWeighted<DyArgumentType, spin + 1>{dy_argument};
  }
};

/*!
 * \brief Specialization for the spin-weighted derivative \f$\eth \bar{\eth}\f$.
 *
 * \details The implemented equation is:
 *
 * \f[
 * \eth \bar{\eth} F = \eth^\prime \bar{\eth}^\prime F
 * - \frac{\eth R \bar{\eth} R}{R^2} (1 - y)^2 \partial_y^2 F
 * - (1 - y)\left(\frac{\eth R}{R} \bar{\eth} \partial_y F
 * + \frac{\bar{\eth} R}{R} \eth \partial_y F
 * + \frac{\eth \bar\eth R}{R} \partial_y F\right),
 * \f]
 *
 * where \f$\eth \bar{\eth}\f$ is the derivative at constant Bondi radius
 * \f$r\f$ and \f$\eth^\prime \bar{\eth}^\prime\f$ is the derivative at constant
 * numerical radius \f$y\f$.
 */
template <typename ArgumentTag>
struct ApplySwshJacobianInplace<Spectral::Swsh::Tags::Derivative<
    ArgumentTag, Spectral::Swsh::Tags::EthEthbar>> {
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags =
      tmpl::list<Tags::OneMinusY, Tags::EthRDividedByR,
                 Tags::EthEthbarRDividedByR>;

  using return_tags = tmpl::list<Spectral::Swsh::Tags::Derivative<
      ArgumentTag, Spectral::Swsh::Tags::EthEthbar>>;
  using argument_tags = tmpl::append<integration_independent_tags>;
  using on_demand_argument_tags =
      tmpl::list<Tags::Dy<ArgumentTag>, Tags::Dy<Tags::Dy<ArgumentTag>>,
                 Spectral::Swsh::Tags::Derivative<Tags::Dy<ArgumentTag>,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<
                     Tags::Dy<ArgumentTag>, Spectral::Swsh::Tags::Ethbar>>;

  static constexpr int spin =
      Spectral::Swsh::Tags::Derivative<ArgumentTag,
                                       Spectral::Swsh::Tags::EthEthbar>::spin;
  template <typename DyArgumentType, typename DyDyArgumentType,
            typename EthDyArgumentType, typename EthbarDyArgumentType>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, spin>>*>
          eth_ethbar_argument,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&
          eth_ethbar_r_divided_by_r,
      const DyArgumentType& dy_argument, const DyDyArgumentType& dy_dy_argument,
      const EthDyArgumentType& eth_dy_argument,
      const EthbarDyArgumentType ethbar_dy_argument) {
    get(*eth_ethbar_argument) -=
        get(eth_r_divided_by_r) * conj(get(eth_r_divided_by_r)) *
            (square(get(one_minus_y)) *
             SpinWeighted<DyDyArgumentType, spin>{dy_dy_argument}) +
        get(one_minus_y) *
            (get(eth_r_divided_by_r) *
                 SpinWeighted<EthbarDyArgumentType, spin - 1>{
                     ethbar_dy_argument} +
             conj(get(eth_r_divided_by_r)) *
                 SpinWeighted<EthDyArgumentType, spin + 1>{eth_dy_argument} +
             get(eth_ethbar_r_divided_by_r) *
                 SpinWeighted<DyArgumentType, spin>{dy_argument});
  }
};

/*!
 * \brief Specialization for the spin-weighted derivative \f$\bar{\eth} \eth\f$.
 *
 * \details The implemented equation is:
 *
 * \f[
 * \bar{\eth} \eth F = \bar{\eth}^\prime \eth^\prime F
 * - \frac{\eth R \bar{\eth} R}{R^2} (1 - y)^2 \partial_y^2 F
 * - (1 - y)\left(\frac{\eth R}{R} \bar{\eth} \partial_y F
 * + \frac{\bar{\eth} R}{R} \eth \partial_y F
 * + \frac{\eth \bar\eth R}{R} \partial_y F\right),
 * \f]
 *
 * where \f$\bar{\eth} \eth\f$ is the derivative at constant Bondi radius
 * \f$r\f$ and \f$\bar{\eth}^\prime \eth^\prime\f$ is the derivative at constant
 * numerical radius \f$y\f$.
 */
template <typename ArgumentTag>
struct ApplySwshJacobianInplace<Spectral::Swsh::Tags::Derivative<
    ArgumentTag, Spectral::Swsh::Tags::EthbarEth>> {
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags =
      tmpl::list<Tags::OneMinusY, Tags::EthRDividedByR,
                 Tags::EthEthbarRDividedByR>;

  using return_tags = tmpl::list<Spectral::Swsh::Tags::Derivative<
      ArgumentTag, Spectral::Swsh::Tags::EthbarEth>>;
  using argument_tags = tmpl::append<integration_independent_tags>;
  using on_demand_argument_tags =
      tmpl::list<Tags::Dy<ArgumentTag>, Tags::Dy<Tags::Dy<ArgumentTag>>,
                 Spectral::Swsh::Tags::Derivative<Tags::Dy<ArgumentTag>,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<
                     Tags::Dy<ArgumentTag>, Spectral::Swsh::Tags::Ethbar>>;

  static constexpr int spin =
      Spectral::Swsh::Tags::Derivative<ArgumentTag,
                                       Spectral::Swsh::Tags::EthbarEth>::spin;
  template <typename DyArgumentType, typename DyDyArgumentType,
            typename EthDyArgumentType, typename EthbarDyArgumentType>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, spin>>*>
          ethbar_eth_argument,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&
          eth_ethbar_r_divided_by_r,
      const DyArgumentType& dy_argument, const DyDyArgumentType& dy_dy_argument,
      const EthDyArgumentType& eth_dy_argument,
      const EthbarDyArgumentType ethbar_dy_argument) {
    get(*ethbar_eth_argument) -=
        get(eth_r_divided_by_r) * conj(get(eth_r_divided_by_r)) *
            (square(get(one_minus_y)) *
             SpinWeighted<DyDyArgumentType, spin>{dy_dy_argument}) +
        get(one_minus_y) *
            (get(eth_r_divided_by_r) *
                 SpinWeighted<EthbarDyArgumentType, spin - 1>{
                     ethbar_dy_argument} +
             conj(get(eth_r_divided_by_r)) *
                 SpinWeighted<EthDyArgumentType, spin + 1>{eth_dy_argument} +
             get(eth_ethbar_r_divided_by_r) *
                 SpinWeighted<DyArgumentType, spin>{dy_argument});
  }
};

/*!
 * \brief Specialization for the spin-weighted derivative \f$\eth \eth\f$.
 *
 * \details The implemented equation is:
 *
 * \f[
 * \eth \eth F = \eth^\prime \eth^\prime F
 * - (1 - y)^2 \frac{(\eth R)^2}{R^2} \partial_y^2 F
 * - (1 - y) \left( 2 \frac{\eth R}{R} \eth \partial_y F
 * + \frac{\eth \eth R}{R} \partial_y F\right),
 * \f]
 *
 * where \f$\eth \eth\f$ is the derivative at constant Bondi radius \f$r\f$ and
 * \f$\eth^\prime \eth^\prime\f$ is the derivative at constant numerical radius
 * \f$y\f$.
 */
template <typename ArgumentTag>
struct ApplySwshJacobianInplace<Spectral::Swsh::Tags::Derivative<
    ArgumentTag, Spectral::Swsh::Tags::EthEth>> {
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags =
      tmpl::list<Tags::OneMinusY, Tags::EthRDividedByR,
                 Tags::EthEthRDividedByR>;

  using return_tags = tmpl::list<Spectral::Swsh::Tags::Derivative<
      ArgumentTag, Spectral::Swsh::Tags::EthEth>>;
  using argument_tags = tmpl::append<integration_independent_tags>;
  using on_demand_argument_tags =
      tmpl::list<Tags::Dy<ArgumentTag>, Tags::Dy<Tags::Dy<ArgumentTag>>,
                 Spectral::Swsh::Tags::Derivative<Tags::Dy<ArgumentTag>,
                                                  Spectral::Swsh::Tags::Eth>>;

  static constexpr int spin =
      Spectral::Swsh::Tags::Derivative<ArgumentTag,
                                       Spectral::Swsh::Tags::EthEth>::spin;
  template <typename DyArgumentType, typename DyDyArgumentType,
            typename EthDyArgumentType>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, spin>>*>
          eth_eth_argument,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_eth_r_divided_by_r,
      const DyArgumentType& dy_argument, const DyDyArgumentType& dy_dy_argument,
      const EthDyArgumentType& eth_dy_argument) {
    get(*eth_eth_argument) -=
        square(get(eth_r_divided_by_r)) *
            (square(get(one_minus_y)) *
             SpinWeighted<DyDyArgumentType, spin - 2>{dy_dy_argument}) +
        get(one_minus_y) *
            (2.0 * get(eth_r_divided_by_r) *
                 SpinWeighted<EthDyArgumentType, spin - 1>{eth_dy_argument} +
             get(eth_eth_r_divided_by_r) *
                 SpinWeighted<DyArgumentType, spin - 2>{dy_argument});
  }
};

/*!
 * \brief Specialization for the spin-weighted derivative \f$\bar{\eth}
 * \bar{\eth}\f$.
 *
 * \details The implemented equation is:
 *
 * \f[
 * \bar{\eth} \bar{\eth} F = \bar{\eth}^\prime \bar{\eth}^\prime F
 * - (1 - y)^2 \frac{(\bar{\eth} R)^2}{R^2} \partial_y^2 F
 * - (1 - y) \left( 2 \frac{\bar{\eth} R}{R} \bar{\eth} \partial_y F
 * + \frac{\bar{\eth} \bar{\eth} R}{R} \partial_y F\right),
 * \f]
 *
 * where \f$\bar{\eth} \bar{\eth}\f$ is the derivative at constant Bondi radius
 * \f$r\f$ and \f$\bar{\eth}^\prime \bar{\eth}^\prime\f$ is the derivative at
 * constant numerical radius \f$y\f$.
 */
template <typename ArgumentTag>
struct ApplySwshJacobianInplace<Spectral::Swsh::Tags::Derivative<
    ArgumentTag, Spectral::Swsh::Tags::EthbarEthbar>> {
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags =
      tmpl::list<Tags::OneMinusY, Tags::EthRDividedByR,
                 Tags::EthEthRDividedByR>;

  using return_tags = tmpl::list<Spectral::Swsh::Tags::Derivative<
      ArgumentTag, Spectral::Swsh::Tags::EthbarEthbar>>;
  using argument_tags = tmpl::append<integration_independent_tags>;
  using on_demand_argument_tags =
      tmpl::list<Tags::Dy<ArgumentTag>, Tags::Dy<Tags::Dy<ArgumentTag>>,
                 Spectral::Swsh::Tags::Derivative<
                     Tags::Dy<ArgumentTag>, Spectral::Swsh::Tags::Ethbar>>;

  static constexpr int spin = Spectral::Swsh::Tags::Derivative<
      ArgumentTag, Spectral::Swsh::Tags::EthbarEthbar>::spin;
  template <typename DyArgumentType, typename DyDyArgumentType,
            typename EthbarDyArgumentType>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, spin>>*>
          ethbar_ethbar_argument,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_eth_r_divided_by_r,
      const DyArgumentType& dy_argument, const DyDyArgumentType& dy_dy_argument,
      const EthbarDyArgumentType& ethbar_dy_argument) {
    get(*ethbar_ethbar_argument) -=
        square(conj(get(eth_r_divided_by_r))) *
            (square(get(one_minus_y)) *
             SpinWeighted<DyDyArgumentType, spin + 2>{dy_dy_argument}) +
        get(one_minus_y) *
            (2.0 * conj(get(eth_r_divided_by_r)) *
                 SpinWeighted<EthbarDyArgumentType, spin + 1>{
                     ethbar_dy_argument} +
             conj(get(eth_eth_r_divided_by_r)) *
                 SpinWeighted<DyArgumentType, spin + 2>{dy_argument});
  }
};

namespace detail {
// A helper to forward to the `ApplySwshJacobianInplace` mutators that takes
// advantage of the `OnDemandInputsForSwshJacobian`, which computes blaze
// template expressions for those quantities for which it is anticipated to be
// sufficiently cheap to repeatedly compute that it is worth saving the cost of
// additional storage (e.g. conjugates and derivatives for which the product
// rule applies)
template <typename DerivativeTag, typename DataBoxTagList,
          typename... OnDemandTags>
void apply_swsh_jacobian_helper(
    const gsl::not_null<db::DataBox<DataBoxTagList>*> box,
    tmpl::list<OnDemandTags...> /*meta*/) noexcept {
  db::mutate_apply<ApplySwshJacobianInplace<DerivativeTag>>(
      box, OnDemandInputsForSwshJacobian<OnDemandTags>{}(*box)...);
}
}  // namespace detail

/*!
 * \brief This routine evaluates the set of inputs to the CCE integrand for
 * `BondiValueTag` which are spin-weighted angular derivatives.

 * \details This function is called on the \ref DataBoxGroup holding the
 * relevant CCE data during each hypersurface integration step, after evaluating
 * `mutate_all_pre_swsh_derivatives_for_tag()` with template argument
 * `BondiValueTag` and before evaluating `ComputeBondiIntegrand<BondiValueTag>`.
 * Provided a \ref DataBoxGroup with the appropriate tags (including
 * `Cce::all_pre_swsh_derivative_tags`, `Cce::all_swsh_derivative_tags`,
 * `Cce::all_transform_buffer_tags`,  `Cce::pre_computation_tags`, and
 * `Cce::Tags::LMax`), this function will apply all of the necessary
 * mutations to update
 * `Cce::single_swsh_derivative_tags_to_compute_for<BondiValueTag>` and
 * `Cce::second_swsh_derivative_tags_to_compute_for<BondiValueTag>` to their
 * correct values for the current values of the remaining (input) tags.
 */
template <typename BondiValueTag, typename DataBoxTagList>
void mutate_all_swsh_derivatives_for_tag(
    const gsl::not_null<db::DataBox<DataBoxTagList>*> box) noexcept {
  // The collection of spin-weighted derivatives cannot be applied as individual
  // compute items, because it is better to aggregate similar spins and dispatch
  // to libsharp in groups. So, we supply a bulk mutate operation which takes in
  // multiple Variables from the presumed DataBox, and alters their values as
  // necessary.
  db::mutate_apply<Spectral::Swsh::AngularDerivatives<
      single_swsh_derivative_tags_to_compute_for_t<BondiValueTag>>>(box);
  tmpl::for_each<single_swsh_derivative_tags_to_compute_for_t<BondiValueTag>>(
      [&box](auto derivative_tag_v) noexcept {
        using derivative_tag = typename decltype(derivative_tag_v)::type;
        detail::apply_swsh_jacobian_helper<derivative_tag>(
            box, typename ApplySwshJacobianInplace<
                     derivative_tag>::on_demand_argument_tags{});
      });

  db::mutate_apply<Spectral::Swsh::AngularDerivatives<
      second_swsh_derivative_tags_to_compute_for_t<BondiValueTag>>>(box);
  tmpl::for_each<second_swsh_derivative_tags_to_compute_for_t<BondiValueTag>>(
      [&box](auto derivative_tag_v) noexcept {
        using derivative_tag = typename decltype(derivative_tag_v)::type;
        detail::apply_swsh_jacobian_helper<derivative_tag>(
            box, typename ApplySwshJacobianInplace<
                     derivative_tag>::on_demand_argument_tags{});
      });
}
}  // namespace Cce
