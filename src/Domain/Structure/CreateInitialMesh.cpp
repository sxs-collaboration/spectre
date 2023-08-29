// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/CreateInitialMesh.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Quadrature.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::Initialization {
template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const ElementId<Dim>& element_id,
    const SpatialDiscretization::Quadrature quadrature,
    const OrientationMap<Dim>& orientation) {
  const auto& unoriented_extents = initial_extents[element_id.block_id()];
  Index<Dim> extents;
  for (size_t i = 0; i < Dim; ++i) {
    extents[i] = gsl::at(unoriented_extents, orientation(i));
  }
  return {extents.indices(), SpatialDiscretization::Basis::Legendre,
          quadrature};
}
}  // namespace domain::Initialization

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                              \
  template Mesh<DIM(data)>                                \
  domain::Initialization::create_initial_mesh<DIM(data)>( \
      const std::vector<std::array<size_t, DIM(data)>>&,  \
      const ElementId<DIM(data)>&,                        \
      const SpatialDiscretization::Quadrature quadrature, \
      const OrientationMap<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
