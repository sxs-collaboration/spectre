// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

class DataVector;

namespace GeneralizedHarmonic {
namespace Tags {
/*!
 * \brief Conjugate momentum to the spacetime metric.
 *
 * \details If \f$ \psi_{ab} \f$ is the spacetime metric, and \f$ N \f$ and
 * \f$ N^i \f$ are the lapse and shift respectively, then we define
 * \f$ \Pi_{ab} = -\frac{1}{N} ( \partial_t \psi_{ab} + N^{i} \phi_{iab} ) \f$
 * where \f$\phi_{iab}\f$ is the variable defined by the tag Phi.
 */
template <size_t Dim, typename Frame>
struct Pi : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "Pi"; }
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * spacetime metric
 * \details If \f$\psi_{ab}\f$ is the spacetime metric then we define
 * \f$\phi_{iab} = \partial_i \psi_{ab}\f$
 */
template <size_t Dim, typename Frame>
struct Phi : db::SimpleTag {
  using type = tnsr::iaa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "Phi"; }
};

struct ConstraintGamma0 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ConstraintGamma0"; }
};
struct ConstraintGamma1 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ConstraintGamma1"; }
};
struct ConstraintGamma2 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ConstraintGamma2"; }
};

/*!
 * \brief Gauge source function for the generalized harmonic system.
 *
 * \details In the generalized / damped harmonic gauge, unlike the simple
 * harmonic gauge, the right hand side of the gauge equation
 * \f$ \square x_a = H_a\f$ is sourced by non-vanishing functions. This variable
 * stores those functions \f$ H_a\f$.
 */
template <size_t Dim, typename Frame>
struct GaugeH : db::SimpleTag {
  using type = tnsr::a<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "GaugeH"; }
};

/*!
 * \brief Spacetime derivatives of the gauge source function for the
 * generalized harmonic system.
 *
 * \details In the generalized / damped harmonic gauge, the right hand side of
 * the gauge equation \f$ \square x_a = H_a\f$ is sourced by non-vanishing
 * functions \f$ H_a\f$. This variable stores their spacetime derivatives
 * \f$ \partial_b H_a\f$.
 */
template <size_t Dim, typename Frame>
struct SpacetimeDerivGaugeH : db::SimpleTag {
  using type = tnsr::ab<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "SpacetimeDerivGaugeH"; }
};

/*!
 * \brief Initial value of the gauge source function for the generalized
 * harmonic system.
 *
 * \details In the generalized / damped harmonic gauge, unlike the simple
 * harmonic gauge, the right hand side of the gauge equation
 * \f$ \square x_a = H_a\f$ is sourced by non-vanishing functions. This variable
 * stores the initial or starting value of those functions \f$ H_a\f$, which
 * are set by the user (based on the choice of initial data) to begin evolution.
 */
template <size_t Dim, typename Frame>
struct InitialGaugeH : db::SimpleTag {
  using type = tnsr::a<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "InitialGaugeH"; }
};

/*!
 * \brief Initial spacetime derivatives of the gauge source function
 * for the generalized harmonic system.
 *
 * \details In the generalized / damped harmonic gauge, the right hand side of
 * the gauge equation \f$ \square x_a = H_a\f$ is sourced by non-vanishing
 * functions \f$ H_a\f$. This variable stores the initial or starting value of
 * the spacetime derivatives of those functions \f$ \partial_b H_a\f$, which
 * are set by the user (based on the choice of initial data) to begin evolution.
 */
template <size_t Dim, typename Frame>
struct SpacetimeDerivInitialGaugeH : db::SimpleTag {
  using type = tnsr::ab<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "SpacetimeDerivInitialGaugeH"; }
};

// @{
/// \brief Tags corresponding to the characteristic fields of the generalized
/// harmonic system.
///
/// \details For details on how these are defined and computed, see
/// CharacteristicSpeedsCompute
template <size_t Dim, typename Frame>
struct UPsi : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "UPsi"; }
};
template <size_t Dim, typename Frame>
struct UZero : db::SimpleTag {
  using type = tnsr::iaa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "UZero"; }
};
template <size_t Dim, typename Frame>
struct UPlus : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "UPlus"; }
};
template <size_t Dim, typename Frame>
struct UMinus : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "UMinus"; }
};
// @}

template <size_t Dim, typename Frame>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 4>;
  static std::string name() noexcept { return "CharacteristicSpeeds"; }
};

template <size_t Dim, typename Frame>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<UPsi<Dim, Frame>, UZero<Dim, Frame>,
                                    UPlus<Dim, Frame>, UMinus<Dim, Frame>>>;
  static std::string name() noexcept { return "CharacteristicFields"; }
};

template <size_t Dim, typename Frame>
struct EvolvedFieldsFromCharacteristicFields : db::SimpleTag {
  using type =
      Variables<tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame, DataVector>,
                           Pi<Dim, Frame>, Phi<Dim, Frame>>>;
  static std::string name() noexcept {
    return "EvolvedFieldsFromCharacteristicFields";
  }
};
}  // namespace Tags

namespace OptionTags {
/*!
 * \ingroup OptionTagsGroup
 * \brief Gauge control parameter determining when to start rolling-on the
 * evolution gauge.
 *
 * \details The evolution gauge is gradually transitioned to (or
 * *rolled-on* to) at the beginning of an evolution. This parameter sets
 * the coordinate time at which roll-on begins.
 */
struct GaugeHRollOnStartTime : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "GaugeHRollOnStartT"; }
  static constexpr OptionString help{
      "Simulation time to start rolling-on evolution gauge"};
};

/*!
 * \ingroup OptionTagsGroup
 * \brief Gauge control parameter determining how long the transition to
 * the evolution gauge should take at the start of an evolution.
 *
 * \details The evolution gauge is gradually transitioned to (or
 * *rolled-on* to) at the beginning of an evolution. This parameter sets
 * the width of the coordinate time window during which roll-on happens.
 */
struct GaugeHRollOnTimeWindow : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "GaugeHRollOnTWindow"; }
  static constexpr OptionString help{
      "Duration of gauge roll-on in simulation time"};
};

/*!
 * \ingroup OptionTagsGroup
 * \brief Gauge control parameter to specify the spatial weighting function
 * that enters damped harmonic gauge source function.
 *
 * \details The evolution gauge source function is multiplied by a spatial
 * weight function which controls where and how much it sources the damped
 * harmonic gauge equation. The weight function is:
 * \f$ W(x^i) = \exp[-(r/\sigma_r)^2] \f$.
 * This weight function can be written with an extra factor inside the exponent
 * in literature, e.g. \cite Deppe2018uye. We will absorb it here in
 * \f$\sigma_r\f$. The parameter this tag tags is \f$ \sigma_r \f$.
 */
template <typename Frame>
struct GaugeHSpatialWeightDecayWidth : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "GaugeHDecayWidth"; }
  static constexpr OptionString help{
      "Spatial width of weighting factor in evolution gauge"};
};
}  // namespace OptionTags
}  // namespace GeneralizedHarmonic
