// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

class DataVector;

namespace gr {
namespace Tags {
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeMetric;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct InverseSpacetimeMetric;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpatialMetric;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct DetAndInverseSpatialMetric;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct InverseSpatialMetric;
template <typename DataType>
struct DetSpatialMetric;
template <typename DataType>
struct SqrtDetSpatialMetric;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct DerivDetSpatialMetric;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct DerivInverseSpatialMetric;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct Shift;
template <typename DataType>
struct Lapse;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct DerivativesOfSpacetimeMetric;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeChristoffelFirstKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeChristoffelSecondKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpatialChristoffelFirstKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpatialChristoffelSecondKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeNormalOneForm;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpacetimeNormalVector;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct TraceSpacetimeChristoffelFirstKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct TraceSpacetimeChristoffelSecondKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct TraceSpatialChristoffelFirstKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct TraceSpatialChristoffelSecondKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpatialChristoffelSecondKindContracted;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct ExtrinsicCurvature;
template <typename DataType>
struct TraceExtrinsicCurvature;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpatialRicci;
template <typename DataType>
struct SpatialRicciScalar;
template <typename DataType>
struct Psi4Real;
template <typename DataType>
struct EnergyDensity;
template <typename DataType>
struct StressTrace;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct MomentumDensity;
template <typename DataType>
struct HamiltonianConstraint;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct MomentumConstraint;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct WeylElectric;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct WeylMagnetic;
template <typename DataType>
struct WeylElectricScalar;
template <typename DataType>
struct WeylMagneticScalar;
}  // namespace Tags
}  // namespace gr
