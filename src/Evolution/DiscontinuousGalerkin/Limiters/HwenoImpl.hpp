// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags.hpp"       // IWYU pragma: keep
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoGridHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t VolumeDim>
class Element;
template <size_t VolumeDim>
class ElementId;
template <size_t>
class Mesh;
/// \endcond

namespace Limiters {
namespace Weno_detail {

// Caching class that holds various precomputed terms used in the constrained-
// fit algebra on each element.
//
// The terms to precompute and cache depend on the configuration of neighbors to
// the element (how many neighbors, in which directions, of what resolution).
// Each instance of the caching class represents one particular neighbor
// configuration, and typical simulations will create many instances of
// the caching class: one instance for each neighbor-element configuration
// present in the computational domain.
//
// Because the current implementation of the HWENO fitting makes the simplifying
// restrictions of no h/p-refinement, the structure of the caching is also
// simplified. In particular, with no h/p-refinement, the allowable neighbor
// configurations satisfy,
// 1. the element has at most one neighbor per dimension, AND
// 2. the mesh on every neighbor is the same as on the element.
// With these restrictions, the number of independent configurations to be
// cached is greatly reduced. Because an element has 2*VolumeDim boundaries,
// it has 2^(2*VolumeDim) possible configurations of internal/external
// boundaries, and therefore there are 2^(2*VolumeDim) configurations to cache.
// Different elements with the same configuration of internal/external
// boundaries vs. direction can share the same caching-class instance.
//
// Each instance of the caching class holds several terms, some of which also
// depend on the neighbor configuration. The restriction of no h/p-refinement
// therefore simplifies each cache instance, as well as reducing the necessary
// number of instances.
//
// The most complicated term to cache is the A^{-1} matrix. The complexity
// arises because the matrix can take many values depending on (runtime) choices
// made for each individual HWNEO fit: which element is the primary neighbor,
// and which element(s) are the excluded neighbors.
// Fits with one excluded neighbor are overwhelmingly the most likely, and the
// cache is designed for this particular case. Fits with no excluded neighbors
// can arise in elements with only one neighbor (always the primary neighbor);
// these cases are also handled. However, the very rare case in which more than
// one neighbor is excluded is not handled by the cache; the caller must compute
// A^{-1} from scratch if this scenario arises.
template <size_t VolumeDim>
class ConstrainedFitCache {
 public:
  ConstrainedFitCache(const Element<VolumeDim>& element,
                      const Mesh<VolumeDim>& mesh) noexcept;

  // Valid calls must satisfy these constraints on directions_to_exclude:
  // - the vector is empty, OR
  // - the vector contains one element, and this element is a direction
  //   different from the direction to the primary neighbor.
  // The very rare case where more than one neighbor is excluded from the fit is
  // not cached. In this case, A^{-1} must be computed from scratch.
  const Matrix& retrieve_inverse_a_matrix(
      const Direction<VolumeDim>& primary_direction,
      const std::vector<Direction<VolumeDim>>& directions_to_exclude) const
      noexcept;

  DataVector quadrature_weights;
  DirectionMap<VolumeDim, Matrix> interpolation_matrices;
  DirectionMap<VolumeDim, DataVector>
      quadrature_weights_dot_interpolation_matrices;
  // The many possible values of A^{-1} are stored in a map of maps. The outer
  // map indexes over the primary neighbor, and the inner map indexes over the
  // excluded neighbor.
  // This data structure is not perfect: configurations with only one neighbor
  // (always the primary neighbor) lead to no excluded neighbors, and there is
  // no natural place to hold A^{-1} in the maps. For this case, we simply store
  // the data in the normally-nonsensical slot where
  // excluded_neighbor == primary_neighbor.
  DirectionMap<VolumeDim, DirectionMap<VolumeDim, Matrix>> inverse_a_matrices;
};

// Return the appropriate cache for the given element and mesh.
template <size_t VolumeDim>
const ConstrainedFitCache<VolumeDim>& constrained_fit_cache(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh) noexcept;

// Compute the inverse of the matrix A_st for the constrained fit.
// See the documentation of `hweno_modified_neighbor_solution`.
template <size_t VolumeDim>
Matrix inverse_a_matrix(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const DataVector& quadrature_weights,
    const DirectionMap<VolumeDim, Matrix>& interpolation_matrices,
    const DirectionMap<VolumeDim, DataVector>&
        quadrature_weights_dot_interpolation_matrices,
    const Direction<VolumeDim>& primary_direction,
    const std::vector<Direction<VolumeDim>>& directions_to_exclude) noexcept;

// Not all secondary neighbors (i.e., neighbors that are not the primary) are
// included in the HWENO constrained fit. In particular, for the tensor
// component specified by `Tag` and `tensor_index`, the secondary neighbor whose
// mean is most different from the troubled cell's mean is excluded.
//
// Zhu2016 indicate that when multiple secondary neighbors share the property of
// being "most different" from the troubled cell, then all of these secondary
// neighbors should be excluded from the minimization. The probability of this
// occuring is exceedingly low, but we handle it anyway. We return the excluded
// secondary neighbors in a vector.
//
// Note that if there is only one neighbor, it is the primary neighbor, and so
// there are no secondary neighbors to exclude. We return an empty vector. This
// scenario can arise in various test cases, but is unlikely to arise in science
// cases.
template <typename Tag, size_t VolumeDim, typename Package>
std::vector<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>
secondary_neighbors_to_exclude_from_fit(
    const double local_mean, const size_t tensor_index,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, Package,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>&
        primary_neighbor) noexcept {
  // Rare case: with only one neighbor, there is no secondary to exclude
  if (UNLIKELY(neighbor_data.size() == 1)) {
    return std::vector<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>{};
  }

  // Identify element with maximum mean difference
  const auto mean_difference = [&tensor_index, &local_mean ](
      const std::pair<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
                      Package>& neighbor_and_data) noexcept {
    return fabs(
        get<::Tags::Mean<Tag>>(neighbor_and_data.second.means)[tensor_index] -
        local_mean);
  };

  double max_difference = std::numeric_limits<double>::lowest();
  std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>
      neighbor_max_difference{};
  for (const auto& neighbor_and_data : neighbor_data) {
    const auto& neighbor = neighbor_and_data.first;
    if (neighbor == primary_neighbor) {
      continue;
    }
    const double difference = mean_difference(neighbor_and_data);
    if (difference > max_difference) {
      max_difference = difference;
      neighbor_max_difference = neighbor;
    }
  }

  std::vector<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>
      neighbors_to_exclude{{neighbor_max_difference}};

  // See if other elements share this maximum mean difference. This loop should
  // only rarely find other neighbors with the same maximal mean difference to
  // add to the vector, so it will usually not change the vector.
  for (const auto& neighbor_and_data : neighbor_data) {
    const auto& neighbor = neighbor_and_data.first;
    if (neighbor == primary_neighbor or neighbor == neighbor_max_difference) {
      continue;
    }
    const double difference = mean_difference(neighbor_and_data);
    if (UNLIKELY(equal_within_roundoff(difference, max_difference))) {
      neighbors_to_exclude.push_back(neighbor);
    }
  }

  ASSERT(not alg::found(neighbors_to_exclude, primary_neighbor),
         "Logical inconsistency: trying to exclude the primary neighbor.");

  return neighbors_to_exclude;
}

// Compute the vector b_s for the constrained fit. For details, see the
// documentation of `hweno_modified_neighbor_solution` below.
template <typename Tag, size_t VolumeDim, typename Package>
DataVector b_vector(
    const Mesh<VolumeDim>& mesh, const size_t tensor_index,
    const DataVector& quadrature_weights,
    const DirectionMap<VolumeDim, Matrix>& interpolation_matrices,
    const DirectionMap<VolumeDim, DataVector>&
        quadrature_weights_dot_interpolation_matrices,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, Package,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>&
        primary_neighbor,
    const std::vector<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>&
        neighbors_to_exclude) noexcept {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  DataVector b(number_of_grid_points, 0.);

  for (const auto& neighbor_and_data : neighbor_data) {
    // Generally, neighbors_to_exclude contains just one neighbor to exclude,
    // and the loop over neighbor_data will hit this (in 3D) roughly 1/6 times.
    // So the condition is somewhat, but not overwhelmingly, unlikely:
    if (UNLIKELY(alg::found(neighbors_to_exclude, neighbor_and_data.first))) {
      continue;
    }

    const auto& direction = neighbor_and_data.first.first;
    ASSERT(interpolation_matrices.contains(direction),
           "interpolation_matrices does not contain key: " << direction);
    ASSERT(
        quadrature_weights_dot_interpolation_matrices.contains(direction),
        "quadrature_weights_dot_interpolation_matrices does not contain key: "
            << direction);

    const auto& neighbor_mesh = mesh;
    const auto& neighbor_quadrature_weights = quadrature_weights;
    const auto& interpolation_matrix = interpolation_matrices.at(direction);
    const auto& quadrature_weights_dot_interpolation_matrix =
        quadrature_weights_dot_interpolation_matrices.at(direction);

    const auto& neighbor_tensor_component =
        get<Tag>(neighbor_and_data.second.volume_data)[tensor_index];

    // Add terms from the primary neighbor
    if (neighbor_and_data.first == primary_neighbor) {
      for (size_t r = 0; r < neighbor_mesh.number_of_grid_points(); ++r) {
        for (size_t s = 0; s < number_of_grid_points; ++s) {
          b[s] += neighbor_tensor_component[r] *
                  neighbor_quadrature_weights[r] * interpolation_matrix(r, s);
        }
      }
    }
    // Add terms from the secondary neighbors
    else {
      const double quadrature_weights_dot_u = [&]() noexcept {
        double result = 0.;
        for (size_t r = 0; r < neighbor_mesh.number_of_grid_points(); ++r) {
          result +=
              neighbor_tensor_component[r] * neighbor_quadrature_weights[r];
        }
        return result;
      }
      ();
      b += quadrature_weights_dot_u *
           quadrature_weights_dot_interpolation_matrix;
    }
  }
  return b;
}

// Solve the constrained fit problem that gives the HWENO modified solution,
// for one particular tensor component. For details, see documentation of
// `hweno_modified_neighbor_solution` below.
template <typename Tag, size_t VolumeDim, typename Package>
void solve_constrained_fit(
    const gsl::not_null<DataVector*> constrained_fit_result,
    const DataVector& u, const size_t tensor_index,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, Package,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>&
        primary_neighbor,
    const std::vector<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>&
        neighbors_to_exclude) noexcept {
  ASSERT(not alg::found(neighbors_to_exclude, primary_neighbor),
         "Logical inconsistency: trying to exclude the primary neighbor.");
  ASSERT(not neighbors_to_exclude.empty() or neighbor_data.size() == 1,
         "The HWENO constrained fit algorithm expects at least one neighbor \n"
         "to exclude from the fit (unless if the element has a single \n"
         "neighbor, which would automatically be the primary neighbor).");

  // Get the cache of linear algebra quantities for this element
  const ConstrainedFitCache<VolumeDim>& cache =
      constrained_fit_cache(element, mesh);

  // Because we don't support h-refinement, the direction is the only piece
  // of the neighbor information that we actually need.
  const Direction<VolumeDim> primary_direction = primary_neighbor.first;
  const std::vector<Direction<VolumeDim>>
      directions_to_exclude = [&neighbors_to_exclude]() noexcept {
    std::vector<Direction<VolumeDim>> result(neighbors_to_exclude.size());
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] = neighbors_to_exclude[i].first;
    }
    return result;
  }
  ();

  const DataVector& w = cache.quadrature_weights;
  const DirectionMap<VolumeDim, Matrix>& interp_matrices =
      cache.interpolation_matrices;
  const DirectionMap<VolumeDim, DataVector>& w_dot_interp_matrices =
      cache.quadrature_weights_dot_interpolation_matrices;

  // Use cache if possible, or compute matrix if we are in edge case
  const Matrix& inverse_a =
      LIKELY(directions_to_exclude.size() < 2)
          ? cache.retrieve_inverse_a_matrix(primary_direction,
                                            directions_to_exclude)
          : inverse_a_matrix(element, mesh, w, interp_matrices,
                             w_dot_interp_matrices, primary_direction,
                             directions_to_exclude);

  const DataVector b = b_vector<Tag>(mesh, tensor_index, w, interp_matrices,
                                     w_dot_interp_matrices, neighbor_data,
                                     primary_neighbor, neighbors_to_exclude);

  const size_t number_of_points = b.size();
  const DataVector inverse_a_times_b = apply_matrices(
      std::array<std::reference_wrapper<const Matrix>, 1>{{inverse_a}}, b,
      Index<1>(number_of_points));
  const DataVector inverse_a_times_w = apply_matrices(
      std::array<std::reference_wrapper<const Matrix>, 1>{{inverse_a}}, w,
      Index<1>(number_of_points));

  // Compute Lagrange multiplier:
  // Note: we take w as an argument (instead of as a lambda capture), because
  //       some versions of Clang incorrectly warn about capturing w.
  const double lagrange_multiplier =
      [&number_of_points, &inverse_a_times_b, &inverse_a_times_w, &
       u ](const DataVector& local_w) noexcept {
    double numerator = 0.;
    double denominator = 0.;
    for (size_t s = 0; s < number_of_points; ++s) {
      numerator += local_w[s] * (inverse_a_times_b[s] - u[s]);
      denominator += local_w[s] * inverse_a_times_w[s];
    }
    return -numerator / denominator;
  }
  (w);

  // Compute solution:
  *constrained_fit_result =
      inverse_a_times_b + lagrange_multiplier * inverse_a_times_w;
}

template <size_t VolumeDim, typename Package>
using LimiterNeighborData = std::unordered_map<
    std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, Package,
    boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>;

/*!
 * \ingroup LimitersGroup
 * \brief Compute the HWENO modified solution for a particular tensor
 * from a particular neighbor element
 *
 * The HWENO limiter reconstructs a new solution from the linear combination of
 * the local DG solution and a "modified" solution from each neighbor element.
 * This function computes the modified solution for a particular tensor and
 * neighbor, following Section 3 of \cite Zhu2016.
 *
 * The modified solution associated with a particular neighbor (the "primary"
 * neighbor) is obtained by solving a constrained fit over the local element,
 * the primary neighbor, and the other ("secondary") neighbors of the local
 * element. This fit seeks to minimize in a least-squared sense:
 * 1. The distance between the modified solution and the original solution on
 *    the primary neighbor.
 * 2. The distance between the cell average of the modified solution and the
 *    cell average of the original solution on each secondary neighbor. Note
 *    however that one secondary neighbor (or, rarely, several) is excluded from
 *    this minimization: the secondary neighbor(s) where the original solution
 *    has the most different cell average from the local element. This helps to
 *    prevent an outlier (e.g., near a shock) from biasing the fit.
 *
 * The constraint on the minimization is the following: the cell average of the
 * modified solution on the local element must equal the cell average of the
 * local element's original solution.
 *
 * Below we give the mathematical form of the constraints described above and
 * show how these are translated into a numerical algorithm.
 *
 * Consider an element \f$I_0\f$ with neighbors \f$I_1, I_2, ...\f$. For a
 * given tensor component \f$u\f$, the values on each of these elements are
 * \f$u^{(0)}, u^{(1)}, u^{(2)}, ...\f$. Taking for the sake of example the
 * primary neighbor to be \f$I_1\f$, the modified solution \f$\phi\f$ must
 * minimize
 *
 * \f[
 * \chi^2 = \int_{I_1} (\phi - u^{(1)})^2 dV
 *          + \sum_{\ell \in L} \left(
 *                              \int_{I_{\ell}} ( \phi - u^{(\ell)} ) dV
 *                              \right)^2,
 * \f]
 *
 * subject to the constaint
 *
 * \f[
 * C = \int_{I_0} ( \phi - u^{(0)} ) dV = 0.
 * \f]
 *
 * where \f$\ell\f$ ranges over a subset \f$L\f$ of the secondary neighbors.
 * \f$L\f$ excludes the one (or more) secondary neighbor(s) where the mean of
 * \f$u\f$ is the most different from the mean of \f$u^{(0)}\f$. Typically, only
 * one secondary neighbor is excluded, so \f$L\f$ contains two fewer neighbors
 * than the total number of neighbors to the element. Note that in 1D, this
 * implies that \f$L\f$ is the empty set; for each modified solution, one
 * neighbor is the primary neighbor and the other is the excluded neighbor.
 *
 * The integrals are evaluated by quadrature. We denote the quadrature weights
 * by \f$w_s\f$ and the values of some data \f$X\f$ at the quadrature
 * nodes by \f$X_s\f$. We use subscripts \f$r,s\f$ to denote quadrature nodes
 * on the neighbor and local elements, respectively. The minimization becomes
 *
 * \f[
 * \chi^2 = \sum_r w^{(1)}_r ( \phi_r - u^{(1)}_r )^2
 *          + \sum_{\ell \in L} \left(
 *                              \sum_r w^{(\ell)}_r ( \phi_r - u^{(\ell)}_r )
 *                              \right)^2,
 * \f]
 *
 * subject to the constraint
 *
 * \f[
 * C = \sum_s w^{(0)}_s ( \phi_s - u^{(0)}_s ) = 0.
 * \f]
 *
 * Note that \f$\phi\f$ is a function defined on the local element \f$I_0\f$,
 * and so is fully represented by its values \f$\phi_s\f$ at the quadrature
 * points on this element. When evaluating \f$\phi\f$ on element \f$I_{\ell}\f$,
 * we obtain the function values \f$\phi_r\f$ by polynomial extrapolation,
 * \f$\phi_r = \sum_s \mathcal{I}^{(\ell)}_{rs} \phi_s\f$, where
 * \f$\mathcal{I}^{(\ell)}_{rs}\f$ is the interpolation/extrapolation matrix
 * that interpolates data defined at grid points \f$x_s\f$ and evaluates it at
 * grid points \f$x_r\f$ of \f$I_{\ell}\f$. Thus,
 *
 * \f[
 * \chi^2 = \sum_r w^{(1)}_r
 *                 \left(
 *                 \sum_s \mathcal{I}^{(1)}_{rs} \phi_s - u^{(1)}_r
 *                 \right)^2
 *          + \sum_{\ell \in L} \left(
 *                              \sum_r w^{(\ell)}_r
 *                                     \left(
 *                                     \sum_s \mathcal{I}^{(\ell)}_{rs} \phi_s
 *                                             - u^{(\ell)}_r
 *                                     \right)
 *                      \right)^2.
 * \f]
 *
 * The solution to this optimization problem is found in the standard way,
 * using a Lagrange multiplier \f$\lambda\f$ to impose the constraint:
 *
 * \f[
 * 0 = \frac{d}{d \phi_s} \left( \chi^2 + \lambda C \right).
 * \f]
 *
 * Working out the differentiation with respect to \f$\phi_s\f$ leads to the
 * linear problem that must be inverted to obtain the solution,
 *
 * \f[
 * 0 = A_{st} \phi_t - b_s - \lambda w^{(0)}_s,
 * \f]
 *
 * where
 *
 * \f{align*}{
 * A_{st} &= \sum_r \left( w^{(1)}_r
 *                  \mathcal{I}^{(1)}_{rs} \mathcal{I}^{(1)}_{rt}
 *                  \right)
 *          + \sum_{\ell \in L} \left(
 *                              \sum_r \left(
 *                                     w^{(\ell)}_r \mathcal{I}^{(\ell)}_{rt}
 *                                     \right)
 *                              \cdot
 *                              \sum_r \left(
 *                                     w^{(\ell)}_r \mathcal{I}^{(\ell)}_{rs}
 *                                     \right)
 *                              \right)
 * \\
 * b_s &= \sum_r \left( w^{(1)}_r u^{(1)}_r \mathcal{I}^{(1)}_{rs} \right)
 *         + \sum_{\ell \in L} \left(
 *                             \sum_r \left( w^{(\ell)}_r u^{(\ell)}_r \right)
 *                             \cdot
 *                             \sum_r \left(
 *                                    w^{(\ell)}_r \mathcal{I}^{(\ell)}_{rs}
 *                                    \right)
 *                             \right).
 * \f}
 *
 * Finally, the solution to the constrained fit is
 *
 * \f{align*}{
 * \lambda &= - \frac{ \sum_s w^{(0)}_s
 *                            \left( (A^{-1})_{st} b_t - u^{(0)}_s \right)
 *              }{ \sum_s w^{(0)}_s (A^{-1})_{st} w^{(0)}_t }
 * \\
 * \phi_s &= (A^{-1})_{st} ( b_t + \lambda w^{(0)}_t ).
 * \f}
 *
 * Note that the matrix \f$A\f$ does not depend on the values of the tensor
 * \f$u\f$, so its inverse \f$A^{-1}\f$ can be precomputed and stored.
 *
 * \warning
 * Note also that the implementation currently does not support h- or
 * p-refinement; this is checked by some assertions. The implementation is
 * untested for grids where elements are curved, and it should not be expected
 * to work in these cases.
 */
template <typename Tag, size_t VolumeDim, typename Package>
void hweno_modified_neighbor_solution(
    const gsl::not_null<db::item_type<Tag>*> modified_tensor,
    const db::const_item_type<Tag>& local_tensor,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const LimiterNeighborData<VolumeDim, Package>& neighbor_data,
    const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>&
        primary_neighbor) noexcept {
  ASSERT(Weno_detail::check_element_has_one_similar_neighbor_in_direction(
             element, primary_neighbor.first),
         "Found some amount of h-refinement; this is not supported");
  alg::for_each(neighbor_data, [&mesh](const auto& neighbor_and_data) noexcept {
    ASSERT(neighbor_and_data.second.mesh == mesh,
           "Found some amount of p-refinement; this is not supported");
  });

  for (size_t tensor_index = 0; tensor_index < local_tensor.size();
       ++tensor_index) {
    const auto& tensor_component = local_tensor[tensor_index];
    const auto neighbors_to_exclude =
        secondary_neighbors_to_exclude_from_fit<Tag>(
            mean_value(tensor_component, mesh), tensor_index, neighbor_data,
            primary_neighbor);
    solve_constrained_fit<Tag>(make_not_null(&(*modified_tensor)[tensor_index]),
                               local_tensor[tensor_index], tensor_index,
                               element, mesh, neighbor_data, primary_neighbor,
                               neighbors_to_exclude);
  }
}

// Implement the HWENO limiter for one tensor
template <typename Tag, size_t VolumeDim, typename PackagedData>
void hweno_impl(
    const gsl::not_null<std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>*>
        modified_neighbor_solution_buffer,
    const gsl::not_null<db::item_type<Tag>*> tensor,
    const double neighbor_linear_weight, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  alg::for_each(neighbor_data, [&element, &
                                mesh ](const auto& neighbor_and_data) noexcept {
    ASSERT(Weno_detail::check_element_has_one_similar_neighbor_in_direction(
               element, neighbor_and_data.first.first),
           "Found some amount of h-refinement; this is not supported");
    ASSERT(neighbor_and_data.second.mesh == mesh,
           "Found some amount of p-refinement; this is not supported");
  });

  for (size_t tensor_index = 0; tensor_index < tensor->size(); ++tensor_index) {
    const auto& tensor_component = (*tensor)[tensor_index];
    for (const auto& neighbor_and_data : neighbor_data) {
      const auto& primary_neighbor = neighbor_and_data.first;
      const auto neighbors_to_exclude =
          secondary_neighbors_to_exclude_from_fit<Tag>(
              mean_value(tensor_component, mesh), tensor_index, neighbor_data,
              primary_neighbor);

      DataVector& buffer =
          modified_neighbor_solution_buffer->at(primary_neighbor);
      solve_constrained_fit<Tag>(make_not_null(&buffer), tensor_component,
                                 tensor_index, element, mesh, neighbor_data,
                                 primary_neighbor, neighbors_to_exclude);
    }

    // Sum local and modified neighbor polynomials for the WENO reconstruction
    Weno_detail::reconstruct_from_weighted_sum(
        make_not_null(&((*tensor)[tensor_index])), mesh, neighbor_linear_weight,
        *modified_neighbor_solution_buffer,
        Weno_detail::DerivativeWeight::PowTwoEllOverEllFactorial);
  }
}

}  // namespace Weno_detail
}  // namespace Limiters
