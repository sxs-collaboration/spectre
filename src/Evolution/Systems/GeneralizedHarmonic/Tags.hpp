// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

namespace gh {
namespace Tags {
/*!
 * \brief Conjugate momentum to the spacetime metric.
 *
 * \details If \f$ \psi_{ab} \f$ is the spacetime metric, and \f$ N \f$ and
 * \f$ N^i \f$ are the lapse and shift respectively, then we define
 * \f$ \Pi_{ab} = -\frac{1}{N} ( \partial_t \psi_{ab} + N^{i} \Phi_{iab} ) \f$
 * where \f$\Phi_{iab}\f$ is the variable defined by the tag Phi.
 */
template <typename DataType, size_t Dim, typename Frame>
struct Pi : db::SimpleTag {
  using type = tnsr::aa<DataType, Dim, Frame>;
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * spacetime metric
 * \details If \f$\psi_{ab}\f$ is the spacetime metric then we define
 * \f$\Phi_{iab} = \partial_i \psi_{ab}\f$
 */
template <typename DataType, size_t Dim, typename Frame>
struct Phi : db::SimpleTag {
  using type = tnsr::iaa<DataType, Dim, Frame>;
};

/*!
 * \brief Gauge source function for the generalized harmonic system.
 *
 * \details In the generalized / damped harmonic gauge, unlike the simple
 * harmonic gauge, the right hand side of the gauge equation
 * \f$ \square x_a = H_a\f$ is sourced by non-vanishing functions. This variable
 * stores those functions \f$ H_a\f$.
 */
template <typename DataType, size_t Dim, typename Frame>
struct GaugeH : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame>;
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
template <typename DataType, size_t Dim, typename Frame>
struct SpacetimeDerivGaugeH : db::SimpleTag {
  using type = tnsr::ab<DataType, Dim, Frame>;
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
template <typename DataType, size_t Dim, typename Frame>
struct InitialGaugeH : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame>;
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
template <typename DataType, size_t Dim, typename Frame>
struct SpacetimeDerivInitialGaugeH : db::SimpleTag {
  using type = tnsr::ab<DataType, Dim, Frame>;
};

/// @{
/// \brief Tags corresponding to the characteristic fields of the generalized
/// harmonic system.
///
/// \details For details on how these are defined and computed, see
/// CharacteristicSpeedsCompute
template <typename DataType, size_t Dim, typename Frame>
struct VSpacetimeMetric : db::SimpleTag {
  using type = tnsr::aa<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct VZero : db::SimpleTag {
  using type = tnsr::iaa<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct VPlus : db::SimpleTag {
  using type = tnsr::aa<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct VMinus : db::SimpleTag {
  using type = tnsr::aa<DataType, Dim, Frame>;
};
/// @}

template <typename DataType, size_t Dim, typename Frame>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataType, 4>;
};

template <typename DataType, size_t Dim, typename Frame>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<
      VSpacetimeMetric<DataType, Dim, Frame>, VZero<DataType, Dim, Frame>,
      VPlus<DataType, Dim, Frame>, VMinus<DataType, Dim, Frame>>>;
};

template <typename DataType, size_t Dim, typename Frame>
struct EvolvedFieldsFromCharacteristicFields : db::SimpleTag {
  using type = Variables<
      tmpl::list<gr::Tags::SpacetimeMetric<DataType, Dim, Frame>,
                 Pi<DataType, Dim, Frame>, Phi<DataType, Dim, Frame>>>;
};

/*!
 * \brief Tags corresponding to various constraints of the generalized
 * harmonic system, and their diagnostically useful combinations.
 * \details For details on how these are defined and computed, see
 * `GaugeConstraintCompute`, `FConstraintCompute`, `TwoIndexConstraintCompute`,
 * `ThreeIndexConstraintCompute`, `FourIndexConstraintCompute`, and
 * `ConstraintEnergyCompute` respectively
 */
template <typename DataType, size_t SpatialDim, typename Frame>
struct GaugeConstraint : db::SimpleTag {
  using type = tnsr::a<DataType, SpatialDim, Frame>;
};
/// \copydoc GaugeConstraint
template <typename DataType, size_t SpatialDim, typename Frame>
struct FConstraint : db::SimpleTag {
  using type = tnsr::a<DataType, SpatialDim, Frame>;
};
/// \copydoc GaugeConstraint
template <typename DataType, size_t SpatialDim, typename Frame>
struct TwoIndexConstraint : db::SimpleTag {
  using type = tnsr::ia<DataType, SpatialDim, Frame>;
};
/// \copydoc GaugeConstraint
template <typename DataType, size_t SpatialDim, typename Frame>
struct ThreeIndexConstraint : db::SimpleTag {
  using type = tnsr::iaa<DataType, SpatialDim, Frame>;
};
/// \copydoc GaugeConstraint
template <typename DataType, size_t SpatialDim, typename Frame>
struct FourIndexConstraint : db::SimpleTag {
  using type = tnsr::iaa<DataType, SpatialDim, Frame>;
};
/// \copydoc GaugeConstraint
template <typename DataType, size_t SpatialDim, typename Frame>
struct ConstraintEnergy : db::SimpleTag {
  using type = Scalar<DataType>;
};
}  // namespace Tags

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * Groups option tags related to the GeneralizedHarmonic evolution system.
 */
struct Group {
  static std::string name() { return "GeneralizedHarmonic"; }
  static constexpr Options::String help{"Options for the GH evolution system"};
  using group = evolution::OptionTags::SystemGroup;
};
}  // namespace OptionTags
}  // namespace gh
