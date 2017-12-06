// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace GeneralizedHarmonic {
template <size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeMetric;
template <size_t Dim, typename Frame = Frame::Inertial>
struct Pi;
template <size_t Dim, typename Frame = Frame::Inertial>
struct Phi;

template <size_t Dim, typename Frame = Frame::Inertial>
struct InverseSpatialMetric;
template <size_t Dim, typename Frame = Frame::Inertial>
struct Shift;
template <size_t Dim, typename Frame = Frame::Inertial>
struct Lapse;
struct ConstraintGamma0;
struct ConstraintGamma1;
struct ConstraintGamma2;
template <size_t Dim, typename Frame = Frame::Inertial>
struct GaugeH;
template <size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeDerivGaugeH;
template <size_t Dim, typename Frame = Frame::Inertial>
struct InverseSpacetimeMetric;
template <size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeChristoffelFirstKind;
template <size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeNormalOneForm;
template <size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeNormalVector;
template <size_t Dim, typename Frame = Frame::Inertial>
struct TraceSpacetimeChristoffelFirstKind;
}  // namespace GeneralizedHarmonic
