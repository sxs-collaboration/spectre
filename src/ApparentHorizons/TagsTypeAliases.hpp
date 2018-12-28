// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

namespace StrahlkorperTags {
/// Defines type aliases used in Strahlkorper-related Tags.
namespace aliases {
template <typename Frame>
using ThetaPhi = tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>;
template <typename Frame>
using OneForm = tnsr::i<DataVector, 3, Frame>;
template <typename Frame>
using Vector = tnsr::I<DataVector, 3, Frame>;
template <typename Frame>
using Jacobian =
    Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
           index_list<SpatialIndex<3, UpLo::Up, Frame>,
                      SpatialIndex<2, UpLo::Lo, ::Frame::Spherical<Frame>>>>;
template <typename Frame>
using InvJacobian =
    Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
           index_list<SpatialIndex<2, UpLo::Up, ::Frame::Spherical<Frame>>,
                      SpatialIndex<3, UpLo::Lo, Frame>>>;
template <typename Frame>
using InvHessian =
    Tensor<DataVector, tmpl::integral_list<std::int32_t, 3, 2, 1>,
           index_list<SpatialIndex<2, UpLo::Up, ::Frame::Spherical<Frame>>,
                      SpatialIndex<3, UpLo::Lo, Frame>,
                      SpatialIndex<3, UpLo::Lo, Frame>>>;
template <typename Frame>
using SecondDeriv = tnsr::ii<DataVector, 3, Frame>;
}  // namespace aliases
}  // namespace StrahlkorperTags
