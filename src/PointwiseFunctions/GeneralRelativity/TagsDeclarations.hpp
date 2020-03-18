// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

class DataVector;

namespace gr {
namespace Tags {
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpacetimeMetric;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct InverseSpacetimeMetric;

template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpatialMetric;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct DetAndInverseSpatialMetric;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct InverseSpatialMetric;
template <typename DataType = DataVector>
struct DetSpatialMetric;
template <typename DataType = DataVector>
struct SqrtDetSpatialMetric;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct Shift;
template <typename DataType = DataVector>
struct Lapse;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct DerivSpacetimeMetric;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct DerivativesOfSpacetimeMetric;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpacetimeChristoffelFirstKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpacetimeChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpatialChristoffelFirstKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpatialChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpacetimeNormalOneForm;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpacetimeNormalVector;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct TraceSpacetimeChristoffelFirstKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct TraceSpatialChristoffelFirstKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct TraceSpatialChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ExtrinsicCurvature;
template <typename DataType = DataVector>
struct TraceExtrinsicCurvature;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpatialRicci;
template <typename DataType = DataVector>
struct EnergyDensity;
}  // namespace Tags
}  // namespace gr
