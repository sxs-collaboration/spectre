// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/ComplexDataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/ComplexModalVector.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SpinWeighted.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

// IWYU pragma: no_forward_declare SpinWeighted

namespace Spectral {
namespace Swsh {
namespace Tags {

/// \ingroup SwshGroup
/// \brief Struct for labeling the \f$\eth\f$ spin-weighted derivative in tags
struct Eth {};
/// \ingroup SwshGroup
/// \brief Struct for labeling the \f$\bar{\eth}\f$ spin-weighted derivative in
/// tags
struct Ethbar {};
/// \ingroup SwshGroup
/// \brief Struct for labeling the \f$\eth^2\f$ spin-weighted derivative in tags
struct EthEth {};
/// \ingroup SwshGroup
/// \brief Struct for labeling the \f$\bar{\eth} \eth\f$ spin-weighted
/// derivative in tags
struct EthbarEth {};
/// \ingroup SwshGroup
/// \brief Struct for labeling the \f$\eth \bar{\eth}\f$ spin-weighted
/// derivative in tags
struct EthEthbar {};
/// \ingroup SwshGroup
/// \brief Struct for labeling the \f$\bar{\eth}^2\f$ spin-weighted derivative
/// in tags
struct EthbarEthbar {};
/// \brief Struct which acts as a placeholder for a spin-weighted derivative
/// label in the spin-weighted derivative utilities, but represents a 'no-op':
/// no derivative is taken.
struct NoDerivative {};

namespace detail {

// utility function for determining the change of spin after a spin-weighted
// derivative has been applied.
template <typename DerivativeKind>
constexpr int derivative_spin_weight_adjustment = 0;

template <>
constexpr int derivative_spin_weight_adjustment<Eth> = 1;
template <>
constexpr int derivative_spin_weight_adjustment<Ethbar> = -1;
template <>
constexpr int derivative_spin_weight_adjustment<EthEth> = 2;
template <>
constexpr int derivative_spin_weight_adjustment<EthbarEthbar> = -2;

// The below tags are used to find the new type represented by the spin-weighted
// derivative of a spin-weighted quantity. The derivatives alter the spin
// weights, and so the utility `adjust_spin_weight_t<Tag, DerivativeKind>` is a
// metafunction that determines the correct spin-weighted type for the
// spin-weighted derivative `DerivativeKind` of `Tag`.
//
// if there is no `Tag::type::spin`, but the internal type is a compatible
// complex vector type, the spin associated with `Tag` is assumed to be 0.
template <typename DataType, typename DerivativeKind>
struct adjust_spin_weight {
  using type =
      Scalar<SpinWeighted<DataType,
                          derivative_spin_weight_adjustment<DerivativeKind>>>;
};

// case for if there is a `Tag::type::spin`
template <typename DataType, int Spin, typename DerivativeKind>
struct adjust_spin_weight<SpinWeighted<DataType, Spin>, DerivativeKind> {
  using type = Scalar<SpinWeighted<
      DataType, Spin + derivative_spin_weight_adjustment<DerivativeKind>>>;
};

template <typename Tag, typename DerivativeKind>
using adjust_spin_weight_t =
    typename adjust_spin_weight<typename Tag::type::type, DerivativeKind>::type;

// Helper function for creating an appropriate prefix tag name from one of the
// spin-weighted derivative types and an existing tag name
template <typename DerivativeKind>
std::string compose_spin_weighted_derivative_name(
    const std::string& suffix) noexcept;

}  // namespace detail

/// \ingroup SwshGroup
/// \brief Prefix tag representing the spin-weighted derivative of a
/// spin-weighted scalar.
///
/// Template Parameters:
/// - `Tag`: The tag to prefix
/// - `DerivativeKind`: The type of spin-weighted derivative. This may be any of
///   the labeling structs: `Eth`, `Ethbar`, `EthEth`, `EthbarEth`, `EthEthbar`,
///   `EthbarEthbar`, or `NoDerivative`.
///
/// Type Aliases and static values:
/// - `type`: Always a `SpinWeighted<Scalar<ComplexDataVector>, Spin>`, where
///   `Spin` is the spin weight after the derivative `DerivativeKind` has been
///   applied.
/// - `tag`: An alias to the wrapped tag `Tag`. Provided for applicability to
///   general `db::PrefixTag` functionality.
/// - `derived_from`: Another alias to the wrapped tag `Tag`. Provided so that
///   utilities that use this prefix for taking derivatives can have a more
///   expressive code style.
/// - `derivative_kind`: Type alias to `DerivativeKind`, represents the kind of
///   spin-weighted derivative applied to `Tag`
/// - `spin`: The spin weight of the scalar after the derivative has been
///   applied.
template <typename Tag, typename DerivativeKind>
struct Derivative : db::PrefixTag, db::SimpleTag {
  using type = detail::adjust_spin_weight_t<Tag, DerivativeKind>;
  using tag = Tag;
  // derived_from is provided as a second alias so that utilities that assume a
  // derivative prefix tag fail earlier during compilation
  using derived_from = Tag;
  using derivative_kind = DerivativeKind;
  const static int spin = type::type::spin;
  static std::string name() noexcept {
    return detail::compose_spin_weighted_derivative_name<DerivativeKind>(
        Tag::name());
  }
};

/// \ingroup SwshGroup
/// \brief Prefix tag representing the spin-weighted spherical harmonic
/// transform of a spin-weighted scalar.
///
/// Template Parameters:
/// - `Tag`: The tag to prefix.
///
/// Type aliases and static values:
/// - `type`: Always a `SpinWeighted<Scalar<ComplexModalVector>, Spin>`, where
///   `Spin` is the same spin weight of the pre-transformed `Tag`, and 0 if
///   provided a `Tag` with a `Type` that is not `SpinWeighted`
/// - `tag`: An alias to the wrapped tag `Tag`. Provided for applicability to
///   general `db::PrefixTag` functionality
/// - `transform_of`: Another alias to the wrapped tag `Tag`. Provided so that
///   utilities that use this prefix for taking transforms can have a more
///   expressive code style.
template <typename Tag>
struct SwshTransform : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<
      ComplexModalVector,
      detail::adjust_spin_weight_t<Tag, NoDerivative>::type::spin>>;
  using tag = Tag;
  // transform_of is provided as a second alias so that utilities that assume a
  // derivative prefix tag fail earlier during compilation
  using transform_of = Tag;
  const static int spin = type::type::spin;
  static std::string name() noexcept {
    return "SwshTransform(" + Tag::name() + ")";
  }
};
}  // namespace Tags

namespace detail {
// implementation for get_tags_with_spin
template <typename Tag, typename S>
struct has_spin : cpp17::bool_constant<Tag::type::type::spin == S::value> {};

template <typename PrefixTag, typename S>
struct wrapped_has_spin : has_spin<typename PrefixTag::tag, S> {};

}  // namespace detail

/// \ingroup SwshGroup
/// \brief Extract from `TagList` the subset of those tags that have a static
/// int member `spin` equal to the template parameter `Spin`.
///
/// \snippet Test_SwshTags.cpp get_tags_with_spin
template <int Spin, typename TagList>
using get_tags_with_spin = tmpl::remove_duplicates<tmpl::filter<
    TagList, detail::has_spin<tmpl::_1, std::integral_constant<int, Spin>>>>;

/// \ingroup SwshGroup
/// \brief Extract from `TagList` the subset of  those tags that wrap a tag
/// that has a static int member `spin` equal to the template parameter `Spin`.
///
/// \snippet Test_SwshTags.cpp get_prefix_tags_that_wrap_tags_with_spin
template <int Spin, typename TagList>
using get_prefix_tags_that_wrap_tags_with_spin =
    tmpl::filter<TagList, tmpl::bind<detail::wrapped_has_spin, tmpl::_1,
                                     std::integral_constant<int, Spin>>>;

}  // namespace Swsh
}  // namespace Spectral
