// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/optional.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"

/// \cond
namespace domain {
class BlockId;
}  // namespace domain
class DataVector;
template <size_t VolumeDim>
class Domain;
template <typename IdType, typename DataType>
class IdPair;
/// \endcond

/// \ingroup ComputationalDomainGroup
///
/// Computes the block logical coordinates and the containing `BlockId` of
/// a set of points, given coordinates in a particular frame.
///
/// \details Returns a std::vector<boost::optional<IdPair<BlockId,coords>>>,
/// where the vector runs over the points and is indexed in the same order as
/// the input coordinates `x`. For each point, the `IdPair` holds the
/// block logical coords of that point and the `BlockId` of the `Block` that
/// contains that point.
/// The boost::optional is empty if the point is not in any Block.
/// If a point is on a shared boundary of two or more `Block`s, it is
/// returned only once, and is considered to belong to the `Block`
/// with the smaller `BlockId`.
template <size_t Dim, typename Frame>
auto block_logical_coordinates(
    const Domain<Dim>& domain, const tnsr::I<DataVector, Dim, Frame>& x,
    double time = std::numeric_limits<double>::signaling_NaN(),
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = std::unordered_map<
            std::string,
            std::unique_ptr<
                domain::FunctionsOfTime::FunctionOfTime>>{}) noexcept
    -> std::vector<boost::optional<
        IdPair<domain::BlockId, tnsr::I<double, Dim, ::Frame::Logical>>>>;
