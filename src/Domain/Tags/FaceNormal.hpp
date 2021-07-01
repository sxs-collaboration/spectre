// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"

namespace domain::Tags {

/// The _normalized_ face normal
template <size_t Dim, typename LocalFrame = Frame::Inertial>
using FaceNormal =
    ::Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim, LocalFrame>>;

/// The magnitude of the _unnormalized_ face normal, see
/// `::unnormalized_face_normal`
template <size_t Dim, typename LocalFrame = Frame::Inertial>
using UnnormalizedFaceNormalMagnitude =
    ::Tags::Magnitude<Tags::UnnormalizedFaceNormal<Dim, LocalFrame>>;

}  // namespace domain::Tags
