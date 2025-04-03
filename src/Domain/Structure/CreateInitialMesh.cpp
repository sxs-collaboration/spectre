// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/CreateInitialMesh.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {
template <size_t Dim>
std::array<Spectral::Basis, Dim> make_basis(
    const std::array<domain::Topology, Dim>& topologies) {
  const auto topology_to_basis = [](const domain::Topology topology) {
    switch (topology) {
      case (domain::Topology::I1):
        return Spectral::Basis::Legendre;
      case (domain::Topology::S1):
        return Spectral::Basis::Fourier;
      case (domain::Topology::S2Colatitude):
        [[fallthrough]];
      case (domain::Topology::S2Longitude):
        return Spectral::Basis::SphericalHarmonic;
      case (domain::Topology::B2Radial):
        [[fallthrough]];
      case (domain::Topology::B2Angular):
        return Spectral::Basis::B2Marcus;
      default:
        ERROR("Invalid topology");
    }
  };
  return map_array(topologies, topology_to_basis);
}

template <size_t Dim>
std::array<Spectral::Quadrature, Dim> make_quadrature(
    const std::array<domain::Topology, Dim>& topologies,
    const Spectral::Quadrature legendre_quadrature) {
  const auto topology_to_quadrature =
      [&legendre_quadrature](const domain::Topology topology) {
        switch (topology) {
          case (domain::Topology::I1):
            return legendre_quadrature;
          // NOLINTNEXTLINE(bugprone-branch-clone)
          case (domain::Topology::S1):
            [[fallthrough]];
          case (domain::Topology::S2Longitude):
            [[fallthrough]];
          case (domain::Topology::B2Angular):
            return Spectral::Quadrature::Equiangular;
          case (domain::Topology::S2Colatitude):
            return Spectral::Quadrature::Gauss;
          case (domain::Topology::B2Radial):
            return Spectral::Quadrature::GaussRadauUpper;
          default:
            ERROR("Invalid topology");
        }
      };
  return map_array(topologies, topology_to_quadrature);
}

template <size_t Dim>
bool is_radially_refined_b2(const std::array<domain::Topology, Dim>& topologies,
                            const ElementId<Dim>& element_id) {
  const auto it = alg::find(topologies, domain::Topology::B2Radial);
  if (it == topologies.end()) {
    return false;
  }
  return gsl::at(element_id.refinement_levels(),
                 std::distance(topologies.begin(), it)) != 0;
}
}  // namespace

namespace domain::Initialization {
template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Element<Dim>& element,
    const Spectral::Quadrature legendre_quadrature) {
  return {initial_extents[element.id().block_id()],
          make_basis(element.topologies()),
          make_quadrature(element.topologies(), legendre_quadrature)};
}

template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Block<Dim>& block, [[maybe_unused]] const ElementId<Dim>& element_id,
    const Spectral::Quadrature legendre_quadrature) {
  ASSERT(not is_radially_refined_b2(block.topologies(), element_id),
         "Splitting Topology::B2Radial is not yet supported");
  return {initial_extents[block.id()], make_basis(block.topologies()),
          make_quadrature(block.topologies(), legendre_quadrature)};
}
}  // namespace domain::Initialization

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template Mesh<DIM(data)>                                                   \
  domain::Initialization::create_initial_mesh<DIM(data)>(                    \
      const std::vector<std::array<size_t, DIM(data)>>& initial_extents,     \
      const Element<DIM(data)>& element,                                     \
      const Spectral::Quadrature legendre_quadrature);                       \
  template Mesh<DIM(data)>                                                   \
  domain::Initialization::create_initial_mesh<DIM(data)>(                    \
      const std::vector<std::array<size_t, DIM(data)>>& initial_extents,     \
      const Block<DIM(data)>& block, const ElementId<DIM(data)>& element_id, \
      const Spectral::Quadrature legendre_quadrature);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
