// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

namespace gh {

/// \brief Tags for the generalized harmonic formulation of Einstein equations
namespace Tags {
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct Pi;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct Phi;

template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct InitialGaugeH;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeDerivInitialGaugeH;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct GaugeH;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeDerivGaugeH;

template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct VSpacetimeMetric;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct VZero;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct VPlus;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct VMinus;

template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct CharacteristicSpeeds;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct CharacteristicFields;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct EvolvedFieldsFromCharacteristicFields;

template <typename DataType, size_t SpatialDim,
          typename Frame = Frame::Inertial>
struct GaugeConstraint;
template <typename DataType, size_t SpatialDim,
          typename Frame = Frame::Inertial>
struct FConstraint;
template <typename DataType, size_t SpatialDim,
          typename Frame = Frame::Inertial>
struct TwoIndexConstraint;
template <typename DataType, size_t SpatialDim,
          typename Frame = Frame::Inertial>
struct ThreeIndexConstraint;
template <typename DataType, size_t SpatialDim,
          typename Frame = Frame::Inertial>
struct FourIndexConstraint;
template <typename DataType, size_t SpatialDim,
          typename Frame = Frame::Inertial>
struct ConstraintEnergy;
}  // namespace Tags

/// \brief Input option tags for the generalized harmonic evolution system
namespace OptionTags {
struct Group;
}  // namespace OptionTags
}  // namespace gh
