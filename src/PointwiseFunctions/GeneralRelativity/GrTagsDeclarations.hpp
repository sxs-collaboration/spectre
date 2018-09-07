// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

template <typename T>
class DataVectorImpl;
using DataVector = DataVectorImpl<double>;

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
struct InverseSpatialMetric;
template <size_t Dim, typename Frame, typename DataType>
struct SqrtDetSpatialMetric;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct Shift;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct Lapse;

template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpacetimeChristoffelFirstKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpacetimeChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpacetimeNormalOneForm;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpacetimeNormalVector;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct TraceSpacetimeChristoffelFirstKind;
template <size_t Dim, typename Frame, typename DataType>
struct TraceSpatialChristoffelSecondKind;
template <size_t Dim, typename Frame, typename DataType>
struct ExtrinsicCurvature;
template <size_t Dim, typename Frame, typename DataType>
struct TraceExtrinsicCurvature;
}  // namespace Tags
}  // namespace gr
