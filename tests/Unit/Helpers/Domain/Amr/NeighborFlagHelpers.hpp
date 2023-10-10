// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

/// \cond
namespace amr {
enum class Flag;
template <size_t VolumeDim>
struct Info;
}  // namespace amr
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Neighbors;
/// \endcond

namespace TestHelpers::amr {
template <size_t Dim>
using neighbor_info_t = std::unordered_map<ElementId<Dim>, ::amr::Info<Dim>>;

template <size_t Dim>
using valid_info_t = std::vector<neighbor_info_t<Dim>>;

/// Returns all permutations of valid flags for the `neighbors` (in a single
/// direction) of an Element with `element_id` and `element_flags`
template <size_t Dim>
valid_info_t<Dim> valid_neighbor_info(
    const ElementId<Dim>& element_id,
    const std::array<::amr::Flag, Dim>& element_flags,
    const Neighbors<Dim>& neighbors);
}  // namespace TestHelpers::amr
