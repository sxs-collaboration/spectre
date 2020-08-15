// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce {

/// \cond
struct MetricWorldtubeDataManager;
struct AnalyticBoundaryDataManager;
template <typename ToInterpolate, typename Tag>
struct ScriPlusInterpolationManager;
/// \endcond

/// Tags for Cauchy Characteristic Extraction routines
namespace Tags {

// Bondi parameter tags

/// Bondi parameter \f$\beta\f$
struct BondiBeta : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

/// Bondi parameter \f$J\f$
struct BondiJ : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 2>>;
  static std::string name() noexcept { return "J"; }
};

}  // namespace Tags
}  // namespace Cce

namespace Tags {
/// \cond
template <>
struct dt<Cce::Tags::BondiJ> : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "H"; }
  using type = Scalar<::SpinWeighted<ComplexDataVector, 2>>;
  using tag = Cce::Tags::BondiJ;
};
/// \endcond
}  // namespace Tags

namespace Cce {
namespace Tags {
/// \brief Bondi parameter \f$H = \partial_u J\f$.
/// \note The notation in the literature is not consistent regarding this
/// quantity, or whether it is denoted by an \f$H\f$ at all. The SpECTRE CCE
/// module consistently uses it to describe the (retarded) partial time
/// derivative of \f$J\f$ at fixed compactified radius \f$y\f$ (to be contrasted
/// with the physical Bondi radius, which is not directly used for numerical
/// grids).
using BondiH = ::Tags::dt<BondiJ>;

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

/// The surface quantity of Bondi \f$U\f$ evaluated at the null spacetime
/// boundary \f$\mathcal I^+\f$
struct BondiUAtScri : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 1>>;
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
};

/// The derivative with respect to Bondi \f$r\f$
template <typename Tag>
struct Dr : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// The derivative with respect to Bondi retarded time \f$u\f$
template <typename Tag>
struct Du : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
};

/// The spin-weight 2 angular Jacobian factor
struct GaugeC : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 2>>;
};

/// The spin-weight 0 angular Jacobian factor
struct GaugeD : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

/// The conformal factor associated with an angular transformation
struct GaugeOmega : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct News : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -2>>;
};

// For expressing the Cauchy angular coordinates for the worldtube data in terms
// of the evolution angular coordinates.
struct CauchyAngularCoords : db::SimpleTag {
  using type = tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>;
};

// For expressing the Cauchy Cartesian coordinates for the worldtube data in
// terms of the evolution angular coordinates.
struct CauchyCartesianCoords : db::SimpleTag {
  using type = tnsr::i<DataVector, 3>;
};

/// The asymptotically inertial retarded time in terms of the evolution time
/// variable
struct InertialRetardedTime : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Represents \f$\eth u_{\rm inertial}\f$, which is a useful quantity for
/// asymptotic coordinate transformations.
struct EthInertialRetardedTime : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 1>>;
};

/// Complex storage form for the asymptotically inertial retarded time, for
/// taking spin-weighted derivatives
struct ComplexInertialRetardedTime : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

// prefix tags associated with the integrands which are used as input to solvers
// for the CCE equations

/// A prefix tag representing a quantity that will appear on the right-hand side
/// of an explicitly regular differential equation
template <typename Tag>
struct Integrand : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
};

/// A prefix tag representing the boundary data for a quantity on the extraction
/// surface.
template <typename Tag>
struct BoundaryValue : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
};

/// A prefix tag representing the gauge-transformed boundary data for a quantity
/// on the extraction surface.
template <typename Tag>
struct EvolutionGaugeBoundaryValue : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
};

/// A prefix tag representing the coefficient of a pole part of the right-hand
/// side of a singular differential equation
template <typename Tag>
struct PoleOfIntegrand : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
};

/// A prefix tag representing the regular part of the right-hand side of a
/// regular differential equation
template <typename Tag>
struct RegularIntegrand : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
};

/// A prefix tag representing a linear factor that acts on `Tag`. To determine
/// the spin weight, It is assumed that the linear factor plays the role of
/// \f$L\f$ in an equation of the form,
/// \f$ (y - 1) \partial_y H + L H + L^\prime \bar{H} = A + (1 - y) B \f$
template <typename Tag>
struct LinearFactor : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  using tag = Tag;
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
};

// Below are additional tags for values which are frequently used in CCE
// calculations, and therefore worth caching

/// Coordinate value \f$(1 - y)\f$, which will be cached and sent to the
/// implementing functions
struct OneMinusY : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

/// A tag for the first time derivative of the worldtube parameter
/// \f$\partial_u R\f$, where \f$R(u, \theta, \phi)\f$ is Bondi
/// radius of the worldtube.
struct DuR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

/// The value \f$\partial_u R / R\f$, where \f$R(u, \theta, \phi)\f$ is Bondi
/// radius of the worldtube.
struct DuRDividedByR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

/// The value \f$\eth R / R\f$, where \f$R(u, \theta, \phi)\f$ is Bondi
/// radius of the worldtube.
struct EthRDividedByR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 1>>;
  using derivative_kind = Spectral::Swsh::Tags::Eth;
  static constexpr int spin = 1;
};

/// The value \f$\eth \eth R / R\f$, where \f$R(u, \theta, \phi)\f$ is Bondi
/// radius of the worldtube.
struct EthEthRDividedByR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 2>>;
  using derivative_kind = Spectral::Swsh::Tags::EthEth;
  static constexpr int spin = 2;
};

/// The value \f$\eth \bar{\eth} R / R\f$, where \f$R(u, \theta, \phi)\f$ is
/// Bondi radius of the worldtube.
struct EthEthbarRDividedByR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  using derivative_kind = Spectral::Swsh::Tags::EthEthbar;
  static constexpr int spin = 0;
};

/// The value \f$\exp(2\beta)\f$.
struct Exp2Beta : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

/// The value \f$ \bar{J} (Q - 2 \eth \beta ) \f$.
struct JbarQMinus2EthBeta : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -1>>;
};

/// The Bondi radius \f$R(u, \theta, \phi)\f$ is of the worldtube.
struct BondiR : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  static std::string name() noexcept { return "R"; }
};

struct EndTime : db::BaseTag {};

struct StartTime : db::BaseTag {};

/// The manager for populating the worldtube \ref DataBoxGroup from an analytic
/// solution `Cce::Solutions::WorldtubeData`
struct AnalyticBoundaryDataManager : db::SimpleTag {
  using type = ::Cce::AnalyticBoundaryDataManager;
};

/// The Weyl scalar \f$\Psi_0\f$
struct Psi0 : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 2>>;
};

/// The Weyl scalar \f$\Psi_1\f$
struct Psi1 : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 1>>;
};

/// The Weyl scalar \f$\Psi_2\f$
struct Psi2 : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

/// The Weyl scalar \f$\Psi_3\f$
struct Psi3 : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -1>>;
};

/// The Weyl scalar \f$\Psi_4\f$
struct Psi4 : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -2>>;
};

/// The gravitational wave strain \f$h\f$
struct Strain : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -2>>;
};

/// A prefix tag representing the time integral of the value it prefixes
template <typename Tag>
struct TimeIntegral : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
};

/// A prefix tag representing the value at \f$\mathcal I^+\f$
template <typename Tag>
struct ScriPlus : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>;
  using tag = Tag;
};

/// A prefix tag representing an additional correction factor necessary to
/// compute the quantity at \f$\mathcal I^+\f$
template <typename Tag>
struct ScriPlusFactor : db::PrefixTag, db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
  using tag = Tag;
};

template <typename ToInterpolate, typename ObservationTag>
struct InterpolationManager : db::SimpleTag {
  using type = ScriPlusInterpolationManager<ToInterpolate, ObservationTag>;
  static std::string name() noexcept {
    return "InterpolationManager(" + db::tag_name<ObservationTag>() + ")";
  }
};
}  // namespace Tags
}  // namespace Cce
