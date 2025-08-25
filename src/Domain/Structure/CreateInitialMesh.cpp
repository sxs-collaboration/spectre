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
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {
template <size_t Dim>
std::array<Spectral::Basis, Dim> make_basis(
    const std::array<domain::Topology, Dim>& topologies,
    const Spectral::Basis i1_basis) {
  const auto topology_to_basis = [&i1_basis](const domain::Topology topology) {
    switch (topology) {
      case (domain::Topology::I1): {
        ASSERT(i1_basis == Spectral::Basis::Legendre or
                   i1_basis == Spectral::Basis::Chebyshev,
               "Invalid I1 Basis: " << i1_basis);
        return i1_basis;
      }
      case (domain::Topology::S1):
        return Spectral::Basis::Fourier;
      case (domain::Topology::S2Colatitude):
        [[fallthrough]];
      case (domain::Topology::S2Longitude):
        return Spectral::Basis::SphericalHarmonic;
      case (domain::Topology::B2Radial):
        [[fallthrough]];
      case (domain::Topology::B2Angular):
        return Spectral::Basis::ZernikeB2;
      // NOLINTNEXTLINE(bugprone-branch-clone)
      case (domain::Topology::B3Radial):
        [[fallthrough]];
      case (domain::Topology::B3Colatitude):
        [[fallthrough]];
      case (domain::Topology::B3Longitude):
        return Spectral::Basis::ZernikeB3;
      default:
        ERROR("Invalid topology");
    }
  };
  return map_array(topologies, topology_to_basis);
}

template <size_t Dim>
std::array<Spectral::Quadrature, Dim> make_quadrature(
    const std::array<domain::Topology, Dim>& topologies,
    const Spectral::Quadrature i1_quadrature) {
  const auto topology_to_quadrature =
      [&i1_quadrature](const domain::Topology topology) {
        switch (topology) {
          case (domain::Topology::I1): {
            ASSERT(i1_quadrature == Spectral::Quadrature::GaussLobatto or
                       i1_quadrature == Spectral::Quadrature::Gauss,
                   "Invalid I1 Quadrature: " << i1_quadrature);
            return i1_quadrature;
          }
          // NOLINTNEXTLINE(bugprone-branch-clone)
          case (domain::Topology::S1):
            [[fallthrough]];
          case (domain::Topology::S2Longitude):
            [[fallthrough]];
          case (domain::Topology::B3Longitude):
            [[fallthrough]];
          case (domain::Topology::B2Angular):
            return Spectral::Quadrature::Equiangular;
          case (domain::Topology::S2Colatitude):
            [[fallthrough]];
          case (domain::Topology::B3Colatitude):
            return Spectral::Quadrature::Gauss;
          case (domain::Topology::B2Radial):
            [[fallthrough]];
          case (domain::Topology::B3Radial):
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

namespace domain {
template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Element<Dim>& element, const Spectral::Basis i1_basis,
    const Spectral::Quadrature i1_quadrature) {
  return {initial_extents[element.id().block_id()],
          make_basis(element.topologies(), i1_basis),
          make_quadrature(element.topologies(), i1_quadrature)};
}

template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Block<Dim>& block, [[maybe_unused]] const ElementId<Dim>& element_id,
    const Spectral::Basis i1_basis, const Spectral::Quadrature i1_quadrature) {
  ASSERT(not is_radially_refined_b2(block.topologies(), element_id),
         "Splitting Topology::B2Radial is not yet supported");
  return {initial_extents[block.id()], make_basis(block.topologies(), i1_basis),
          make_quadrature(block.topologies(), i1_quadrature)};
}
}  // namespace domain

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template Mesh<DIM(data)> domain::create_initial_mesh<DIM(data)>(           \
      const std::vector<std::array<size_t, DIM(data)>>& initial_extents,     \
      const Element<DIM(data)>& element, const Spectral::Basis i1_basis,     \
      const Spectral::Quadrature i1_quadrature);                             \
  template Mesh<DIM(data)> domain::create_initial_mesh<DIM(data)>(           \
      const std::vector<std::array<size_t, DIM(data)>>& initial_extents,     \
      const Block<DIM(data)>& block, const ElementId<DIM(data)>& element_id, \
      const Spectral::Basis i1_basis, const Spectral::Quadrature _quadrature);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
