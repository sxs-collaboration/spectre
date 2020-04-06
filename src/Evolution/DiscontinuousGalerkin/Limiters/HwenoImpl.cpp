// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/HwenoImpl.hpp"

#include <array>
#include <bitset>
#include <cstddef>
#include <exception>
#include <ostream>
#include <vector>

#include "DataStructures/IndexIterator.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"     // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoGridHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {

// Compute the VolumeDim-dimensional quadrature weights, by taking the tensor
// product of the 1D quadrature weights in each logical dimension of the Mesh.
template <size_t VolumeDim>
DataVector volume_quadrature_weights(const Mesh<VolumeDim>& mesh) noexcept {
  std::array<DataVector, VolumeDim> quadrature_weights_1d{};
  for (size_t d = 0; d < VolumeDim; ++d) {
    gsl::at(quadrature_weights_1d, d) =
        Spectral::quadrature_weights(mesh.slice_through(d));
  }

  DataVector result{mesh.number_of_grid_points(), 1.};
  for (IndexIterator<VolumeDim> s(mesh.extents()); s; ++s) {
    result[s.collapsed_index()] = quadrature_weights_1d[0][s()[0]];
    for (size_t d = 1; d < VolumeDim; ++d) {
      result[s.collapsed_index()] *= gsl::at(quadrature_weights_1d, d)[s()[d]];
    }
  }
  return result;
}

// Compute the matrix that interpolates data in VolumeDim dimensions from
// source_mesh to the grid points of target_mesh, by taking the tensor product
// of the 1D interpolation matrices in each logical dimension of the meshes.
//
// Note: typically it is more efficient to interpolate from source_mesh to
// target_mesh in one logical direction at a time, and to avoid computing this
// matrix. However, the HWENO algorithm requires constructing this matrix
// explicitly.
//
// In particular, for this HWENO application:
// - source_mesh is the mesh of local element.
// - target_mesh is the mesh of the neighboring element in direction. Note that
//   there can only be one neighbor in this direction, and this is checked in
//   the call to `neighbor_grid_points_in_local_logical_coordinates`.
// - rectilinear elements are assumed.
template <size_t VolumeDim>
Matrix volume_interpolation_matrix(
    const Mesh<VolumeDim>& source_mesh, const Mesh<VolumeDim>& target_mesh,
    const Element<VolumeDim>& element,
    const Direction<VolumeDim>& direction) noexcept {
  // The grid points of source_mesh and target_mesh must be in the same
  // coordinates to construct the interpolation matrix. Here we get the points
  // of target_mesh in the local logical coordinates.
  const auto target_1d_coords =
      Limiters::Weno_detail::neighbor_grid_points_in_local_logical_coords(
          source_mesh, target_mesh, element, direction);

  const auto interpolation_matrices_1d =
      intrp::RegularGrid<VolumeDim>(source_mesh, target_mesh, target_1d_coords)
          .interpolation_matrices();

  // The 1D interpolation matrices will be empty if there is no need to
  // interpolate in that particular direction (i.e., the interpolation in that
  // direction is identity). This function undoes the optimization by returning
  // elements of an identity matrix if an empty matrix is found.
  const auto matrix_element = [&interpolation_matrices_1d](
      const size_t dim, const size_t r, const size_t s) noexcept {
    const auto& matrix = gsl::at(interpolation_matrices_1d, dim);
    if (matrix.rows() * matrix.columns() == 0) {
      return (r == s) ? 1. : 0.;
    } else {
      return matrix(r, s);
    }
  };

  Matrix result(target_mesh.number_of_grid_points(),
                source_mesh.number_of_grid_points(), 0.);
  for (IndexIterator<VolumeDim> r(target_mesh.extents()); r; ++r) {
    for (IndexIterator<VolumeDim> s(source_mesh.extents()); s; ++s) {
      result(r.collapsed_index(), s.collapsed_index()) =
          matrix_element(0, r()[0], s()[0]);
      for (size_t d = 1; d < VolumeDim; ++d) {
        result(r.collapsed_index(), s.collapsed_index()) *=
            matrix_element(d, r()[d], s()[d]);
      }
    }
  }
  return result;
}

}  // namespace

namespace Limiters {
namespace Weno_detail {

template <size_t VolumeDim>
Matrix inverse_a_matrix(
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const DataVector& quadrature_weights,
    const DirectionMap<VolumeDim, Matrix>& interpolation_matrices,
    const DirectionMap<VolumeDim, DataVector>&
        quadrature_weights_dot_interpolation_matrices,
    const Direction<VolumeDim>& primary_direction,
    const std::vector<Direction<VolumeDim>>& directions_to_exclude) noexcept {
  ASSERT(not alg::found(directions_to_exclude, primary_direction),
         "Logical inconsistency: trying to exclude the primary direction.");
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  Matrix a(number_of_grid_points, number_of_grid_points, 0.);

  // Loop only over directions where there is a neighbor
  const std::vector<Direction<VolumeDim>>
      directions_with_neighbors = [&element]() noexcept {
    std::vector<Direction<VolumeDim>> result;
    for (const auto& dir_and_neighbors : element.neighbors()) {
      result.push_back(dir_and_neighbors.first);
    }
    return result;
  }
  ();

  // Sanity check that directions_to_exclude is consistent with the element
  ASSERT(
      (not directions_to_exclude.empty()) or
          directions_with_neighbors.size() == 1,
      "directions_to_exclude can only be empty if there is a single neighbor");

  for (const auto& dir : directions_with_neighbors) {
    if (alg::found(directions_to_exclude, dir)) {
      continue;
    }

    const auto& neighbor_mesh = mesh;
    const auto& neighbor_quadrature_weights = quadrature_weights;
    const auto& interpolation_matrix = interpolation_matrices.at(dir);
    const auto& weights_dot_interpolation_matrix =
        quadrature_weights_dot_interpolation_matrices.at(dir);

    // Add terms from the primary neighbor
    if (dir == primary_direction) {
      for (size_t r = 0; r < neighbor_mesh.number_of_grid_points(); ++r) {
        for (size_t s = 0; s < number_of_grid_points; ++s) {
          for (size_t t = 0; t < number_of_grid_points; ++t) {
            a(s, t) += neighbor_quadrature_weights[r] *
                       interpolation_matrix(r, s) * interpolation_matrix(r, t);
          }
        }
      }
    }
    // Add terms from the secondary neighbors
    else {
      for (size_t s = 0; s < number_of_grid_points; ++s) {
        for (size_t t = 0; t < number_of_grid_points; ++t) {
          a(s, t) += weights_dot_interpolation_matrix[s] *
                     weights_dot_interpolation_matrix[t];
        }
      }
    }
  }

  // Invert matrix in-place before returning
  try {
    blaze::invert<blaze::asSymmetric>(a);
  } catch (const std::exception& e) {
    ERROR("Blaze matrix inversion failed with exception:\n" << e.what());
  }

  return a;
}

template <size_t VolumeDim>
ConstrainedFitCache<VolumeDim>::ConstrainedFitCache(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh) noexcept
    : quadrature_weights(volume_quadrature_weights(mesh)) {
  // Cache will only store quantities for directions that have neighbors.
  const std::vector<Direction<VolumeDim>>
      directions_with_neighbors = [&element]() noexcept {
    std::vector<Direction<VolumeDim>> result;
    for (const auto& dir_and_neighbors : element.neighbors()) {
      result.push_back(dir_and_neighbors.first);
    }
    return result;
  }
  ();

  for (const auto& dir : directions_with_neighbors) {
    interpolation_matrices[dir] =
        volume_interpolation_matrix(mesh, mesh, element, dir);
    quadrature_weights_dot_interpolation_matrices[dir] = apply_matrices(
        std::array<Matrix, 1>{{trans(interpolation_matrices.at(dir))}},
        quadrature_weights, Index<1>(quadrature_weights.size()));
  }

  for (const auto& primary_dir : directions_with_neighbors) {
    if (directions_with_neighbors.size() == 1) {
      // With a single neighbor, there can be no neighbors to exclude.
      std::vector<Direction<VolumeDim>> nothing_to_exclude{};
      // To reuse the same data structure from the more common `else` branch,
      // here we stick the data into the (normally nonsensical) slot where
      // `dir_to_exclude == primary_dir`.
      inverse_a_matrices[primary_dir][primary_dir] = inverse_a_matrix(
          mesh, element, quadrature_weights, interpolation_matrices,
          quadrature_weights_dot_interpolation_matrices, primary_dir,
          nothing_to_exclude);
    } else {
      // Cache only handles the case of 1 neighbor to exclude.
      for (const auto& dir_to_exclude : directions_with_neighbors) {
        // Skip the nonsensical case where the primary and excluded neighbors
        // are the same.
        if (dir_to_exclude == primary_dir) {
          continue;
        }
        inverse_a_matrices[primary_dir][dir_to_exclude] = inverse_a_matrix(
            mesh, element, quadrature_weights, interpolation_matrices,
            quadrature_weights_dot_interpolation_matrices, primary_dir,
            {{dir_to_exclude}});
      }
    }
  }
}

template <size_t VolumeDim>
const Matrix& ConstrainedFitCache<VolumeDim>::retrieve_inverse_a_matrix(
    const Direction<VolumeDim>& primary_direction,
    const std::vector<Direction<VolumeDim>>& directions_to_exclude) const
    noexcept {
  if (LIKELY(directions_to_exclude.size() == 1)) {
    return inverse_a_matrices.at(primary_direction)
        .at(directions_to_exclude[0]);
  } else if (directions_to_exclude.empty()) {
    return inverse_a_matrices.at(primary_direction).at(primary_direction);
  } else {
    ERROR(
        "Cache misuse error: asked to retrieve a cached A^{-1} matrix for a\n"
        "configuration where multiple neighboring elements are excluded from\n"
        "the HWENO fit. Because this case is so rare, it is not handled by\n"
        "the cache. The caller should check for multiple neighbors being\n"
        "excluded, and, if this occurs, should bypass the cache and compute\n"
        "A^{-1} directly.");
  }
}

namespace {

template <size_t VolumeDim, size_t DummyIndex>
const ConstrainedFitCache<VolumeDim>& constrained_fit_cache_impl(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh) noexcept {
  // For the cache to be valid,
  // - the mesh must always be the same, so we check it below
  // - the element can be different, as long as it has the same configuration
  //   of internal/external boundaries. this is handled in the calling code
  //   because the boundary configuration sets the value of DummyIndex
  static const Mesh<VolumeDim> mesh_for_cached_matrix = mesh;
  ASSERT(mesh_for_cached_matrix == mesh,
         "This call to constrained_fit_cache_impl received a different Mesh\n"
         "than was previously cached, suggesting that multiple meshes are\n"
         "used in the computational domain. This is not (yet) supported.\n"
         "Cached mesh: "
             << mesh_for_cached_matrix
             << "\n"
                "Argument mesh: "
             << mesh);
  static const ConstrainedFitCache<VolumeDim> result(element, mesh);
  return result;
}

template <size_t VolumeDim, size_t... Is>
const ConstrainedFitCache<VolumeDim>& constrained_fit_cache(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    std::index_sequence<Is...> /*dummy_indices*/) noexcept {
  static const std::array<
      const ConstrainedFitCache<VolumeDim>& (*)(const Element<VolumeDim>&,
                                                const Mesh<VolumeDim>&),
      sizeof...(Is)>
      cache{{&constrained_fit_cache_impl<VolumeDim, Is>...}};

  // Use std::bitset to compute an integer based on the configuration of
  // internal/external boundaries to the element. This is sort of like a hash
  // for indexing into the std::array of ConstrainedFitCache's.
  const size_t index_from_boundary_types = [&element]() noexcept {
    std::bitset<2 * VolumeDim> bits;
    for (size_t d = 0; d < VolumeDim; ++d) {
      for (const Side& side : {Side::Lower, Side::Upper}) {
        // Index into bitset
        const size_t bit_index = 2 * d + (side == Side::Lower ? 0 : 1);
        // Is there a neighbor in this direction?
        const Direction<VolumeDim> dir(d, side);
        const bool neighbor_exists =
            (element.neighbors().find(dir) != element.neighbors().end());
        bits[bit_index] = neighbor_exists;
      }
    }
    return static_cast<size_t>(bits.to_ulong());
  }
  ();
  ASSERT(index_from_boundary_types >= 0 and
             index_from_boundary_types < sizeof...(Is),
         "Got index_from_boundary_types = "
             << index_from_boundary_types << ", but expect only "
             << sizeof...(Is) << " configurations");
  return gsl::at(cache, index_from_boundary_types)(element, mesh);
}

}  // namespace

template <size_t VolumeDim>
const ConstrainedFitCache<VolumeDim>& constrained_fit_cache(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh) noexcept {
  return constrained_fit_cache<VolumeDim>(
      element, mesh, std::make_index_sequence<two_to_the(2 * VolumeDim)>{});
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template Matrix inverse_a_matrix<DIM(data)>(                                 \
      const Mesh<DIM(data)>&, const Element<DIM(data)>&, const DataVector&,    \
      const DirectionMap<DIM(data), Matrix>&,                                  \
      const DirectionMap<DIM(data), DataVector>&, const Direction<DIM(data)>&, \
      const std::vector<Direction<DIM(data)>>&) noexcept;                      \
  template class ConstrainedFitCache<DIM(data)>;                               \
  template const ConstrainedFitCache<DIM(data)>& constrained_fit_cache(        \
      const Element<DIM(data)>&, const Mesh<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Weno_detail
}  // namespace Limiters
