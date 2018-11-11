// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace domain {
class BlockId;
}  // namespace domain
class DataVector;
template <size_t VolumeDim, typename TargetFrame>
class Domain;
template <typename IdType, typename DataType>
class IdPair;
/// \endcond

/// \ingroup ComputationalDomainGroup
///
/// Computes the block logical coordinates and the containing `BlockId`
/// of a set of points, given coordinates in the `Frame` frame.
///
/// \details Returns a std::vector<IdPair<BlockId,coords>>, where the
/// vector runs over the points and is indexed in the same order as
/// the input coordinates `x`. For each point, the `IdPair` holds the
/// block logical coords of that point and the `BlockId` of the `Block` that
/// contains that point.
/// If a point is on a shared boundary of two or more `Block`s, it is
/// returned only once, and is considered to belong to the `Block`
/// with the smaller `BlockId`.
template <size_t Dim, typename Frame>
auto block_logical_coordinates(
    const Domain<Dim, Frame>& domain,
    const tnsr::I<DataVector, Dim, Frame>& x) noexcept
    -> std::vector<IdPair<domain::BlockId,
                          tnsr::I<double, Dim, typename ::Frame::Logical>>>;
