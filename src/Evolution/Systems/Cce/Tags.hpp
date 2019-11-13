// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"

namespace Cce {

/// Tags for Cauchy Characteristic Extraction routines
namespace Tags {

// Bondi parameter tags

/// Bondi parameter \f$\beta\f$
struct BondiBeta : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  static std::string name() noexcept { return "BondiBeta"; }
};

/// \brief Bondi parameter \f$H = \partial_u J\f$.
/// \note The notation in the literature is not consistent regarding this
/// quantity, or whether it is denoted by an \f$H\f$ at all. The SpECTRE CCE
/// module consistently uses it to describe the (retarded) partial time
/// derivative of \f$J\f$ at fixed compactified radius \f$y\f$ (to be contrasted
/// with the physical Bondi radius, which is not directly used for numerical
/// grids).
struct BondiH : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 2>>;
  static std::string name() noexcept { return "H"; }
};

/// Bondi parameter \f$J\f$
struct BondiJ : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 2>>;
  static std::string name() noexcept { return "J"; }
};

/// Bondi parameter \f$\bar{J}\f$
struct BondiJbar : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -2>>;
  static std::string name() noexcept { return "Jbar"; }
};

/// Bondi parameter \f$K = \sqrt{1 + J \bar{J}}\f$
struct BondiK : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  static std::string name() noexcept { return "K"; }
};

/// Bondi parameter \f$Q\f$
struct BondiQ : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 1>>;
  static std::string name() noexcept { return "Q"; }
};

/// Bondi parameter \f$\bar{Q}\f$
struct BondiQbar : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -1>>;
  static std::string name() noexcept { return "Qbar"; }
};

/// Bondi parameter \f$U\f$
struct BondiU : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 1>>;
  static std::string name() noexcept { return "U"; }
};

/// Bondi parameter \f$\bar{U}\f$
struct BondiUbar : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -1>>;
  static std::string name() noexcept { return "Ubar"; }
};

/// Bondi parameter \f$W\f$
struct BondiW : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  static std::string name() noexcept { return "W"; }
};

/// The derivative with respect to the numerical coordinate \f$y = 1 - 2R/r\f$,
/// where \f$R(u, \theta, \phi)\f$ is Bondi radius of the worldtube.
template <typename Tag>
struct Dy : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
  static const size_t dimension_to_differentiate = 2;
  static std::string name() noexcept { return "Dy(" + Tag::name() + ")"; }
};

/// The derivative with respect to Bondi \f$r\f$
template <typename Tag>
struct Dr : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
  static std::string name() noexcept { return "Dr(" + Tag::name() + ")"; }
};

/// The derivative with respect to Bondi retarded time \f$u\f$
template <typename Tag>
struct Du : db::PrefixTag, db::SimpleTag {
    using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
    using tag = Tag;
};

// prefix tags associated with the integrands which are used as input to solvers
// for the CCE equations

/// A prefix tag representing a quantity that will appear on the right-hand side
/// of an explicitly regular differential equation
template <typename Tag>
struct Integrand : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
  static std::string name() noexcept {
    return "Integrand(" + Tag::name() + ")";
  }
};

/// A prefix tag representing the boundary data for a quantity on the extraction
/// surface.
template <typename Tag>
struct BoundaryValue : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
  static std::string name() noexcept {
    return "BoundaryValue(" + Tag::name() + ")";
  }
};

/// A prefix tag representing the coefficient of a pole part of the right-hand
/// side of a singular differential equation
template <typename Tag>
struct PoleOfIntegrand : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
  static std::string name() noexcept {
    return "PoleOfIntegrand(" + Tag::name() + ")";
  }
};

/// A prefix tag representing the regular part of the right-hand side of a
/// regular differential equation
template <typename Tag>
struct RegularIntegrand : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
  static std::string name() noexcept {
    return "RegularIntegrand(" + Tag::name() + ")";
  }
};

/// A prefix tag representing a linear factor that acts on `Tag`. To determine
/// the spin weight, It is assumed that the linear factor plays the role of
/// \f$L\f$ in an equation of the form,
/// \f$ (y - 1) \partial_y H + L H + L^\prime \bar{H} = A + (1 - y) B \f$
template <typename Tag>
struct LinearFactor : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  using tag = Tag;
  static std::string name() noexcept {
    return "LinearFactor(" + Tag::name() + ")";
  }
};

/// A prefix tag representing a linear factor that acts on `Tag`. To determine
/// the spin weight, it is assumed that the linear factor plays the role of
/// \f$L^\prime\f$ in an equation of the form,
/// \f$ (y - 1) \partial_y H + L H + L^\prime \bar{H} = A + (1 - y) B \f$
template <typename Tag>
struct LinearFactorForConjugate : db::PrefixTag, db::SimpleTag {
  using type =
      Scalar<SpinWeighted<ComplexDataVector, 2 * Tag::type::type::spin>>;
  using tag = Tag;
  static std::string name() noexcept {
    return "LinearFactorForConjugate(" + Tag::name() + ")";
  }
};

// Below are additional tags for values which are frequently used in CCE
// calculations, and therefore worth caching

/// Coordinate value \f$(1 - y)\f$, which will be cached and sent to the
/// implementing functions
struct OneMinusY : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  static std::string name() noexcept { return "OneMinusY"; }
};

/// A tag for the first time derivative of the worldtube parameter
/// \f$\partial_u R\f$, where \f$R(u, \theta, \phi)\f$ is Bondi
/// radius of the worldtube.
struct DuR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  static std::string name() noexcept { return "DuR"; }
};

/// The value \f$\partial_u R / R\f$, where \f$R(u, \theta, \phi)\f$ is Bondi
/// radius of the worldtube.
struct DuRDividedByR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  static std::string name() noexcept { return "DuRDividedByR"; }
};

/// The value \f$\eth R / R\f$, where \f$R(u, \theta, \phi)\f$ is Bondi
/// radius of the worldtube.
struct EthRDividedByR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 1>>;
  using derivative_kind = Spectral::Swsh::Tags::Eth;
  static constexpr int spin = 1;
  static std::string name() noexcept { return "EthRDividedByR"; }
};

/// The value \f$\eth \eth R / R\f$, where \f$R(u, \theta, \phi)\f$ is Bondi
/// radius of the worldtube.
struct EthEthRDividedByR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 2>>;
  using derivative_kind = Spectral::Swsh::Tags::EthEth;
  static constexpr int spin = 2;
  static std::string name() noexcept { return "EthEthRDividedByR"; }
};

/// The value \f$\eth \bar{\eth} R / R\f$, where \f$R(u, \theta, \phi)\f$ is
/// Bondi radius of the worldtube.
struct EthEthbarRDividedByR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  using derivative_kind = Spectral::Swsh::Tags::EthEthbar;
  static constexpr int spin = 0;
  static std::string name() noexcept { return "EthEthbarRDividedByR"; }
};

/// The value \f$\exp(2\beta)\f$.
struct Exp2Beta : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  static std::string name() noexcept { return "Exp2Beta"; }
};

/// The value \f$ \bar{J} (Q - 2 \eth \beta ) \f$.
struct JbarQMinus2EthBeta : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -1>>;
  static std::string name() noexcept { return "JbarQMinus2EthBeta"; }
};

/// The Bondi radius \f$R(u, \theta, \phi)\f$ is of the worldtube.
struct BondiR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  static std::string name() noexcept { return "R"; }
};
}  // namespace Tags
}  // namespace Cce
