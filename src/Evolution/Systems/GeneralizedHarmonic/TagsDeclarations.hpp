// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

namespace GeneralizedHarmonic {
namespace Tags {
template <size_t Dim, typename Frame = Frame::Inertial>
struct Pi;
template <size_t Dim, typename Frame = Frame::Inertial>
struct Phi;

struct ConstraintGamma0;
struct ConstraintGamma1;
struct ConstraintGamma2;
template <size_t Dim, typename Frame = Frame::Inertial>
struct GaugeH;
template <size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeDerivGaugeH;
template <size_t Dim, typename Frame = Frame::Inertial>
struct DerivSpatialMetric;
template <size_t Dim, typename Frame = Frame::Inertial>
struct TimeDerivSpatialMetric;
template <size_t Dim, typename Frame = Frame::Inertial>
struct DerivLapse;
struct TimeDerivLapse;
template <size_t Dim, typename Frame = Frame::Inertial>
struct DerivShift;
template <size_t Dim, typename Frame = Frame::Inertial>
struct TimeDerivShift;
template <size_t Dim, typename Frame = Frame::Inertial>
struct TraceExtrinsicCurvature;

template<size_t Dim, typename Frame>
struct UPsi;
template<size_t Dim, typename Frame>
struct UZero;
template<size_t Dim, typename Frame>
struct UPlus;
template<size_t Dim, typename Frame>
struct UMinus;

template<size_t Dim, typename Frame>
struct CharacteristicSpeeds;
template<size_t Dim, typename Frame>
struct CharacteristicFields;
template<size_t Dim, typename Frame>
struct EvolvedFieldsFromCharacteristicFields;
}  // namespace Tags
}  // namespace GeneralizedHarmonic
