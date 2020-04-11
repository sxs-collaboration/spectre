// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

namespace GeneralizedHarmonic {

/// \brief Tags for the generalized harmonic formulation of Einstein equations
namespace Tags {
template <size_t Dim, typename Frame = Frame::Inertial>
struct Pi;
template <size_t Dim, typename Frame = Frame::Inertial>
struct Phi;

struct ConstraintGamma0;
struct ConstraintGamma1;
struct ConstraintGamma2;
struct GaugeHRollOnStartTime;
struct GaugeHRollOnTimeWindow;
template <typename Frame>
struct GaugeHSpatialWeightDecayWidth;
template <size_t Dim, typename Frame = Frame::Inertial>
struct InitialGaugeH;
template <size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeDerivInitialGaugeH;
template <size_t Dim, typename Frame = Frame::Inertial>
struct GaugeH;
template <size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeDerivGaugeH;

template <size_t Dim, typename Frame>
struct VSpacetimeMetric;
template <size_t Dim, typename Frame>
struct VZero;
template <size_t Dim, typename Frame>
struct VPlus;
template <size_t Dim, typename Frame>
struct VMinus;

template <size_t Dim, typename Frame>
struct CharacteristicSpeeds;
template <size_t Dim, typename Frame>
struct CharacteristicFields;
template <size_t Dim, typename Frame>
struct EvolvedFieldsFromCharacteristicFields;

template <size_t SpatialDim, typename Frame>
struct GaugeConstraint;
template <size_t SpatialDim, typename Frame>
struct FConstraint;
template <size_t SpatialDim, typename Frame>
struct TwoIndexConstraint;
template <size_t SpatialDim, typename Frame>
struct ThreeIndexConstraint;
template <size_t SpatialDim, typename Frame>
struct FourIndexConstraint;
template <size_t SpatialDim, typename Frame>
struct ConstraintEnergy;
}  // namespace Tags

/// \brief Input option tags for the generalized harmonic evolution system
namespace OptionTags {
struct Group;
struct GaugeGroup;
struct GaugeHRollOnStart;
struct GaugeHRollOnWindow;
template <typename Frame>
struct GaugeHSpatialDecayWidth;
}  // namespace OptionTags
}  // namespace GeneralizedHarmonic
