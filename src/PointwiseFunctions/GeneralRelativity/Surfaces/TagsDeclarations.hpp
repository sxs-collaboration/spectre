// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace StrahlkorperTags {
template <typename Frame>
struct EuclideanAreaElement;
template <typename Frame>
struct EuclideanAreaElementCompute;
template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegral;
struct OneOverOneFormMagnitude;
template <typename DataType, size_t Dim, typename Frame>
struct OneOverOneFormMagnitudeCompute;
template <typename Frame>
struct UnitNormalOneForm;
template <typename Frame>
struct UnitNormalOneFormCompute;
template <typename Frame>
struct UnitNormalVector;
template <typename Frame>
struct UnitNormalVectorCompute;
template <typename Frame>
struct GradUnitNormalOneForm;
template <typename Frame>
struct GradUnitNormalOneFormCompute;
template <typename Frame>
struct ExtrinsicCurvature;
template <typename Frame>
struct ExtrinsicCurvatureCompute;
struct RicciScalar;
template <typename Frame>
struct RicciScalarCompute;
struct MaxRicciScalar;
struct MaxRicciScalarCompute;
struct MinRicciScalar;
struct MinRicciScalarCompute;
template <typename Frame>
struct DimensionfulSpinVector;
template <typename Frame>
struct DimensionfulSpinVectorCompute;
}  // namespace StrahlkorperTags

namespace StrahlkorperGr::Tags {
template <typename Frame>
struct AreaElement;
template <typename Frame>
struct AreaElementCompute;
template <typename IntegrandTag, typename Frame>
struct SurfaceIntegral;
struct Area;
template <typename Frame>
struct AreaCompute;
struct IrreducibleMass;
template <typename Frame>
struct IrreducibleMassCompute;
struct SpinFunction;
template <typename Frame>
struct SpinFunctionCompute;
struct DimensionfulSpinMagnitude;
template <typename Frame>
struct DimensionfulSpinMagnitudeCompute;
struct ChristodoulouMass;
template <typename Frame>
struct ChristodoulouMassCompute;
template <typename Frame>
struct DimensionlessSpinMagnitude;
template <typename Frame>
struct DimensionlessSpinMagnitudeCompute;
template <typename Frame>
struct DimensionfulSpinVector;
template <typename MeasurementFrame, typename MetricDataFrame>
struct DimensionfulSpinVectorCompute;
}  // namespace StrahlkorperGr::Tags
