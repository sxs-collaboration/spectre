// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"

#include <algorithm>  // IWYU pragma: keep // for std::max

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace dg {
template <size_t Dim>
Mesh<Dim> mortar_mesh(const Mesh<Dim>& face_mesh1,
                      const Mesh<Dim>& face_mesh2) noexcept {
  Index<Dim> mortar_extents{};
  for (size_t i = 0; i < Dim; ++i) {
    mortar_extents[i] = std::max(face_mesh1.extents(i), face_mesh2.extents(i));
  }
  return {mortar_extents.indices(), Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)             \
  template Mesh<DIM(data)> mortar_mesh(  \
      const Mesh<DIM(data)>& face_mesh1, \
      const Mesh<DIM(data)>& face_mesh2) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2))

#undef INSTANTIATE
#undef DIM
}  // namespace dg
