// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CreateInitialMesh.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const ElementId<Dim>& element_id,
    const OrientationMap<Dim>& orientation) noexcept {
  const auto& unoriented_extents = initial_extents[element_id.block_id()];
  Index<Dim> extents;
  for (size_t i = 0; i < Dim; ++i) {
    extents[i] = gsl::at(unoriented_extents, orientation(i));
  }
  return {extents.indices(), Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto};
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                               \
  template Mesh<DIM(data)> create_initial_mesh<DIM(data)>( \
      const std::vector<std::array<size_t, DIM(data)>>&,   \
      const ElementId<DIM(data)>&, const OrientationMap<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
