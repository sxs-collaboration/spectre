// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <unordered_map>
#include <utility>

#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class Direction;
template <size_t Dim, typename T>
class DirectionMap;
template <size_t VolumeDim>
class Element;
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim>
class Mesh;

namespace boost {
template <class T>
struct hash;
}  // namespace boost
/// \endcond

namespace Limiters::Tci {

// Implements the TVB troubled-cell indicator from Cockburn1999.
template <size_t VolumeDim>
bool tvb_minmod_indicator(
    gsl::not_null<Minmod_detail::BufferWrapper<VolumeDim>*> buffer,
    double tvb_constant, const DataVector& u, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes);

// Implements the TVB troubled-cell indicator from Cockburn1999 for several
// tensors. Returns true if any component of any tensor needs limiting.
//
// Expects type `PackagedData` to contain, as in Limiters::Minmod:
// - a variable `means` that is a `TaggedTuple<Tags::Mean<Tags>...>`
// - a variable `element_size` that is a `std::array<double, VolumeDim>`
template <size_t VolumeDim, typename PackagedData, typename... Tags>
bool tvb_minmod_indicator(
    const double tvb_constant, const typename Tags::type&... tensors,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) {
  Minmod_detail::BufferWrapper<VolumeDim> buffer(mesh);
  const auto effective_neighbor_sizes =
      Minmod_detail::compute_effective_neighbor_sizes(element, neighbor_data);

  // Ideally, as soon as one component is found that needs limiting, then we
  // would exit the TCI early with return value `true`. But there is no natural
  // way to return early from the pack expansion. So we use this bool to keep
  // track of whether a previous component needed limiting... and if so, then
  // we simply skip any work.
  bool some_component_needs_limiting = false;
  const auto wrap_tci_one_tensor = [&](auto tag, const auto tensor) {
    if (some_component_needs_limiting) {
      // Skip this tensor completely
      return '0';
    }

    for (size_t tensor_storage_index = 0; tensor_storage_index < tensor.size();
         ++tensor_storage_index) {
      const auto effective_neighbor_means =
          Minmod_detail::compute_effective_neighbor_means<decltype(tag)>(
              tensor_storage_index, element, neighbor_data);

      const DataVector& u = tensor[tensor_storage_index];
      const bool component_needs_limiting = tvb_minmod_indicator(
          make_not_null(&buffer), tvb_constant, u, mesh, element, element_size,
          effective_neighbor_means, effective_neighbor_sizes);

      if (component_needs_limiting) {
        some_component_needs_limiting = true;
        // Skip remaining components of this tensor
        return '0';
      }
    }
    return '0';
  };
  expand_pack(wrap_tci_one_tensor(Tags{}, tensors)...);
  return some_component_needs_limiting;
}

}  // namespace Limiters::Tci
