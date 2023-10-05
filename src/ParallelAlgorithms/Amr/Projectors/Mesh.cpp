// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace amr::projectors {
template <size_t Dim>
Mesh<Dim> mesh(const Mesh<Dim>& old_mesh,
               const std::array<amr::Flag, Dim>& flags) {
  std::array<size_t, Dim> new_extents = old_mesh.extents().indices();
  for (size_t d = 0; d < Dim; ++d) {
    if (gsl::at(flags, d) == amr::Flag::IncreaseResolution) {
      ++gsl::at(new_extents, d);
    } else if (gsl::at(flags, d) == amr::Flag::DecreaseResolution) {
      --gsl::at(new_extents, d);
    }
  }
  return {new_extents, old_mesh.basis(), old_mesh.quadrature()};
}

template <size_t Dim>
Mesh<Dim> parent_mesh(const std::vector<Mesh<Dim>>& children_meshes) {
  const auto& parent_quadrature = children_meshes.front().quadrature();
  const auto& parent_basis = children_meshes.front().basis();

  ASSERT(alg::all_of(
             children_meshes,
             [&parent_quadrature, &parent_basis](const Mesh<Dim>& child_mesh) {
               return (child_mesh.quadrature() == parent_quadrature and
                       child_mesh.basis() == parent_basis);
             }),
         "AMR does not currently support joining elements with different "
         "quadratures or bases");

  // loop over each mesh, returning an array containing the max extent in each
  // dimension
  auto parent_extents = std::accumulate(
      std::next(children_meshes.begin()), children_meshes.end(),
      children_meshes.front().extents().indices(),
      [](auto&& extents, const Mesh<Dim>& mesh) {
        alg::transform(extents, mesh.extents().indices(), extents.begin(),
                       [](size_t a, size_t b) { return std::max(a, b); });
        return extents;
      });

  return {parent_extents, parent_basis, parent_quadrature};
}

template <size_t Dim>
Mesh<Dim> new_mesh(
    const Mesh<Dim>& current_mesh, const std::array<Flag, Dim>& flags,
    const Element<Dim>& element,
    const std::unordered_map<ElementId<Dim>, Info<Dim>>& neighbors_info) {
  // If we are joining, the extents of the new mesh in each dimension will be
  // the maximum of that of the element and the joining neighbors
  if (alg::count(flags, Flag::Join) > 0 and not neighbors_info.empty()) {
    const auto joining_neighbors = ids_of_joining_neighbors(element, flags);
    std::vector<Mesh<Dim>> children_meshes;
    children_meshes.reserve(8);
    children_meshes.push_back(mesh(current_mesh, flags));
    for (const auto& [neighbor_id, neighbor_info] : neighbors_info) {
      if (alg::count(joining_neighbors, neighbor_id) > 0) {
        children_meshes.push_back(neighbor_info.new_mesh);
      }
    }
    return parent_mesh(children_meshes);
  }

  return mesh(current_mesh, flags);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                           \
  template Mesh<DIM(data)> mesh(                                       \
      const Mesh<DIM(data)>& old_mesh,                                 \
      const std::array<amr::Flag, DIM(data)>& flags);                  \
  template Mesh<DIM(data)> parent_mesh(                                \
      const std::vector<Mesh<DIM(data)>>& children_meshes);            \
  template Mesh<DIM(data)> new_mesh(                                   \
      const Mesh<DIM(data)>& current_mesh,                             \
      const std::array<Flag, DIM(data)>& flags,                        \
      const Element<DIM(data)>& element,                               \
      const std::unordered_map<ElementId<DIM(data)>, Info<DIM(data)>>& \
          neighbors_info);
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace amr::projectors
