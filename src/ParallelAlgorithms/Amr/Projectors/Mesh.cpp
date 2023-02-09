// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Amr/Flag.hpp"
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
      [](std::array<size_t, Dim>& extents, const Mesh<Dim>& mesh) {
        alg::transform(extents, mesh.extents().indices(), extents.begin(),
                       [](size_t a, size_t b) { return std::max(a, b); });
        return extents;
      });

  return {parent_extents, parent_basis, parent_quadrature};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                          \
  template Mesh<DIM(data)> mesh(                      \
      const Mesh<DIM(data)>& old_mesh,                \
      const std::array<amr::Flag, DIM(data)>& flags); \
  template Mesh<DIM(data)> parent_mesh(               \
      const std::vector<Mesh<DIM(data)>>& children_meshes);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace amr::projectors
