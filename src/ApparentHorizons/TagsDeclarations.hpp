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
}  // namespace StrahlkorperTags

namespace StrahlkorperGr {
namespace Tags {
template <typename Frame>
struct AreaElement;
template <typename IntegrandTag, typename Frame>
struct SurfaceIntegral;
}  // namespace Tags
}  // namespace StrahlkorperGr
