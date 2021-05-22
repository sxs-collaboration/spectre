// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace StrahlkorperTags {

template <typename Frame>
struct Strahlkorper;
template <typename Frame>
struct ThetaPhi;
template <typename Frame>
struct Rhat;
template <typename Frame>
struct Jacobian;
template <typename Frame>
struct InvJacobian;
template <typename Frame>
struct InvHessian;
template <typename Frame>
struct Radius;
template <typename Frame>
struct CartesianCoords;
template <typename Frame>
struct DxRadius;
template <typename Frame>
struct D2xRadius;
template <typename Frame>
struct LaplacianRadius;
template <typename Frame>
struct NormalOneForm;
template <typename Frame>
struct Tangents;
template <typename Frame>
struct EuclideanAreaElement;
template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegral;
struct OneOverOneFormMagnitude;
template <size_t Dim, typename Frame, typename DataType>
struct OneOverOneFormMagnitudeCompute;
template <typename Frame>
struct UnitNormalOneForm;
template <typename Frame>
struct UnitNormalOneFormCompute;
template <typename Frame>
struct UnitNormalVector;
template <typename Frame>
struct UnitNormalVectorCompute;
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

namespace StrahlkorperGr {
namespace Tags {
template <typename Frame>
struct AreaElement;
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

}  // namespace Tags
}  // namespace StrahlkorperGr
