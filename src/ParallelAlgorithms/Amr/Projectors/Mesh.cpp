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
#include "Domain/Structure/Topology.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
size_t extent_delta(const domain::Topology topology) {
  switch (topology) {
    case domain::Topology::Uninitialized:
      ERROR("Uninitialized topology");
    case domain::Topology::I1:
      return 1;
    case domain::Topology::S1:
      return 2;
    case domain::Topology::S2Colatitude:
      return 1;
    case domain::Topology::S2Longitude:
      return 2;
    case domain::Topology::B1Radial:
      [[fallthrough]];
    case domain::Topology::B2Radial:
      return 1;
    case domain::Topology::B2Angular:
      return 4;
    case domain::Topology::B3Radial:
      return 1;
    case domain::Topology::B3Colatitude:
      return 2;
    case domain::Topology::B3Longitude:
      return 4;
    case domain::Topology::CartoonSphere:
      ERROR("Cannot refine topology CartoonSphere");
    case domain::Topology::CartoonCylinder:
      ERROR("Cannot refine topology CartoonCylinder");
    case domain::Topology::HalfS1:
      return 1;
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("An unknown value of domain::Topology was passed");
      // LCOV_EXCL_STOP
  }
}

template <size_t Dim>
Mesh<Dim> new_mesh(const std::array<size_t, Dim>& old_extents,
                   const std::array<amr::Flag, Dim>& flags,
                   const std::array<domain::Topology, Dim>& topologies,
                   const std::array<Spectral::Basis, Dim>& bases,
                   const std::array<Spectral::Quadrature, Dim>& quadratures) {
  std::array<size_t, Dim> new_extents = old_extents;
  for (size_t d = 0; d < Dim; ++d) {
    const auto flag = gsl::at(flags, d);
    if (flag == amr::Flag::IncreaseResolution) {
      gsl::at(new_extents, d) += extent_delta(gsl::at(topologies, d));
    } else if (flag == amr::Flag::DecreaseResolution) {
      gsl::at(new_extents, d) -= extent_delta(gsl::at(topologies, d));
    }
  }

  return {new_extents, bases, quadratures};
}

template <size_t Dim>
bool has_zernike_basis(const Mesh<Dim>& mesh) {
  const auto first_basis = mesh.basis(0);
  return first_basis == Spectral::Basis::ZernikeB1 or
         first_basis == Spectral::Basis::ZernikeB2 or
         first_basis == Spectral::Basis::ZernikeB3;
}
}  // namespace

namespace amr::projectors {
template <size_t Dim>
Mesh<Dim> p_refined_mesh(const Mesh<Dim>& old_mesh,
                         const std::array<amr::Flag, Dim>& flags,
                         const std::array<domain::Topology, Dim>& topologies) {
  return new_mesh(old_mesh.extents().indices(), flags, topologies,
                  old_mesh.basis(), old_mesh.quadrature());
}

template <size_t Dim>
Mesh<Dim> parent_mesh(const std::vector<Mesh<Dim>>& children_meshes) {
  const auto n_zernike = alg::count_if(children_meshes, has_zernike_basis<Dim>);

  auto parent_quadrature = children_meshes.front().quadrature();
  auto parent_basis = children_meshes.front().basis();

  if (n_zernike > 0 and
      static_cast<size_t>(n_zernike) < children_meshes.size()) {
    const auto zernike_it =
        alg::find_if(children_meshes, has_zernike_basis<Dim>);
    parent_quadrature = zernike_it->quadrature();
    parent_basis = zernike_it->basis();
  } else {
    ASSERT(
        alg::all_of(
            children_meshes,
            [&parent_quadrature, &parent_basis](const Mesh<Dim>& child_mesh) {
              return (child_mesh.quadrature() == parent_quadrature and
                      child_mesh.basis() == parent_basis);
            }),
        "AMR does not currently support joining elements with different "
        "quadratures or bases unless there is a single element with a Zernike "
        "basis.  Children meshes were: "
            << children_meshes);
  }

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

  if constexpr (Dim > 1) {
    if (n_zernike > 0) {
      // B2 and B3 topologies require the radial extents to be related to
      // the angular extents based on the radial mode number n
      if (parent_basis[0] == Spectral::Basis::ZernikeB2) {
        const size_t n =
            std::max(parent_extents[0] - 1, (parent_extents[1] - 1) / 4);
        parent_extents[0] = n + 1;
        parent_extents[1] = 4 * n + 1;
      }
      if constexpr (Dim == 3) {
        if (parent_basis[0] == Spectral::Basis::ZernikeB3) {
          const size_t n =
              std::max(parent_extents[0] - 1, (parent_extents[1] - 1) / 2);
          parent_extents[0] = n + 1;
          parent_extents[1] = 2 * n + 1;
          parent_extents[2] = 4 * n + 1;
        }
      }
    }
  }

  return {parent_extents, parent_basis, parent_quadrature};
}

template <size_t Dim>
Mesh<Dim> child_mesh(const Mesh<Dim>& parent_mesh,
                     const ElementId<Dim>& child_id,
                     const std::array<amr::Flag, Dim>& flags,
                     const std::array<domain::Topology, Dim>& topologies) {
  if (has_zernike_basis(parent_mesh) and child_id.segment_id(0).index() == 1) {
    auto new_extents = parent_mesh.extents().indices();
    auto new_bases = parent_mesh.basis();
    auto new_quadratures = parent_mesh.quadrature();
    if (new_extents[0] == 1) {
      ++new_extents[0];
    }
    using ::operator<<;
    if constexpr (Dim == 2) {
      ASSERT(topologies == domain::topologies::disk,
             "Expected a disk, not " << topologies);
      return new_mesh(new_extents, flags, topologies,
                      Spectral::bases::annulus<>,
                      Spectral::quadratures::annulus<>);
    } else if constexpr (Dim == 3) {
      if (topologies == domain::topologies::full_sphere) {
        return new_mesh(new_extents, flags, topologies,
                        Spectral::bases::spherical_shell<>,
                        Spectral::quadratures::spherical_shell<>);
      } else if (parent_mesh.basis(0) == Spectral::Basis::ZernikeB1) {
        new_bases[0] = Spectral::Basis::Legendre;
        new_quadratures[0] = Spectral::Quadrature::GaussLobatto;
        return new_mesh(new_extents, flags, topologies, new_bases,
                        new_quadratures);
      } else {
        ASSERT(parent_mesh.basis(0) == Spectral::Basis::ZernikeB2,
               "Expected full cylinder, not " << topologies);
        new_bases[0] = Spectral::Basis::Legendre;
        new_bases[1] = Spectral::Basis::Fourier;
        new_quadratures[0] = Spectral::Quadrature::GaussLobatto;
        new_quadratures[1] = Spectral::Quadrature::Equiangular;
        return new_mesh(new_extents, flags, topologies, new_bases,
                        new_quadratures);
      }
    } else {
      ERROR("Did not expect Zernike basis in one dimension");
    }
  }
  return new_mesh(parent_mesh.extents().indices(), flags, topologies,
                  parent_mesh.basis(), parent_mesh.quadrature());
}

template <size_t Dim>
Mesh<Dim> new_mesh_to_communicate_to_neighbors(
    const Mesh<Dim>& current_mesh, const std::array<Flag, Dim>& flags,
    const Element<Dim>& element,
    const std::unordered_map<ElementId<Dim>, Info<Dim>>& neighbors_info) {
  // If we are joining, the extents of the new mesh in each dimension will be
  // the maximum of that of the element and the joining neighbors
  if (alg::count(flags, Flag::Join) > 0 and not neighbors_info.empty()) {
    const auto joining_neighbors = ids_of_joining_neighbors(element, flags);
    std::vector<Mesh<Dim>> children_meshes;
    children_meshes.reserve(8);
    children_meshes.push_back(
        p_refined_mesh(current_mesh, flags, element.topologies()));
    for (const auto& [neighbor_id, neighbor_info] : neighbors_info) {
      if (alg::count(joining_neighbors, neighbor_id) > 0) {
        children_meshes.push_back(neighbor_info.new_mesh);
      }
    }
    return parent_mesh(children_meshes);
  }

  return new_mesh(current_mesh.extents().indices(), flags, element.topologies(),
                  current_mesh.basis(), current_mesh.quadrature());
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                           \
  template Mesh<DIM(data)> p_refined_mesh(                             \
      const Mesh<DIM(data)>& old_mesh,                                 \
      const std::array<amr::Flag, DIM(data)>& flags,                   \
      const std::array<domain::Topology, DIM(data)>& topologies);      \
  template Mesh<DIM(data)> parent_mesh(                                \
      const std::vector<Mesh<DIM(data)>>& children_meshes);            \
  template Mesh<DIM(data)> child_mesh(                                 \
      const Mesh<DIM(data)>& parent_mesh,                              \
      const ElementId<DIM(data)>& child_id,                            \
      const std::array<amr::Flag, DIM(data)>& flags,                   \
      const std::array<domain::Topology, DIM(data)>& topologies);      \
  template Mesh<DIM(data)> new_mesh_to_communicate_to_neighbors(       \
      const Mesh<DIM(data)>& current_mesh,                             \
      const std::array<Flag, DIM(data)>& flags,                        \
      const Element<DIM(data)>& element,                               \
      const std::unordered_map<ElementId<DIM(data)>, Info<DIM(data)>>& \
          neighbors_info);
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace amr::projectors
