// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"          // IWYU pragma: keep
#include "Domain/ElementId.hpp"          // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Variables

/// \cond
class DataVector;
template <size_t>
class Mesh;

namespace boost {
template <class T>
struct hash;
}  // namespace boost
/// \endcond

namespace SlopeLimiters {
namespace Weno_detail {

// Compute the WENO oscillation indicator (also called the smoothness indicator)
//
// The oscillation indicator measures the amount of variation in the input data,
// with larger indicator values corresponding to a larger amount of variation
// (either from large monotonic slopes or from oscillations).
//
// Implements an indicator similar to that of Eq. 23 of Dumbser2007, but with
// the necessary adaptations for use on square/cube grids. We favor this
// indicator because it is formulated in the reference coordinates, which we
// use for the WENO reconstruction, and because it lends itself to an efficient
// implementation.
//
// Where (this reference to be added to References.bib later, when it is cited
// from _rendered_ documentation):
// Dumbser2007:
//   Dumbser, M and Kaeser, M
//   Arbitrary high order non-oscillatory finite volume schemes on unstructured
//   meshes for linear hyperbolic systems
//   https://doi.org/10.1016/j.jcp.2006.06.043
template <size_t VolumeDim>
double oscillation_indicator(const DataVector& data,
                             const Mesh<VolumeDim>& mesh) noexcept;

// Compute the unnormalized nonlinear WENO weights. This is a fairly standard
// choice of weights; see e.g., Eq. 3.9 of Zhong2013 or Eq. 3.6 of Zhu2016.
//
// Where (these references to be added to References.bib later, when they are
// cited from _rendered_ documentation):
// Zhong2013
//   Zhong, X and Shu, C-W
//   A simple weighted essentially non-oscillatory limiter for Runge-Kutta
//   discontinuous Galerkin methods
//   https://doi.org/10.1016/j.jcp.2012.08.028
//
// Zhu20016
//   Zhu, J and Zhong, X and Shu, C and Qiu, J
//   Runge-Kutta Discontinuous Galerkin Method with a Simple and Compact Hermite
//   WENO Limiter
//   https://doi.org/10.4208/cicp.070215.200715a
inline double unnormalized_nonlinear_weight(
    const double linear_weight, const double oscillation_indicator) noexcept {
  return linear_weight / square(1.e-6 + oscillation_indicator);
}

// Compute the WENO weighted reconstruction of several polynomials, see e.g.,
// Eq. 4.3 of Zhu2016. This is fairly standard, though different references can
// differ in their choice of oscillation/smoothness indicator.
template <typename Tag, size_t VolumeDim, typename TagsList>
void reconstruct_from_weighted_sum(
    const gsl::not_null<db::item_type<Tag>*> local_tensor,
    const Mesh<VolumeDim>& mesh, const double neighbor_linear_weight,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        Variables<TagsList>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_vars) noexcept {
  for (size_t tensor_storage_index = 0;
       tensor_storage_index < local_tensor->size(); ++tensor_storage_index) {
    auto& local_polynomial = (*local_tensor)[tensor_storage_index];

    // Store linear weights in `local_weights` and `neighbor_weights`
    // These weights will have to be generalized for multiple neighbors per
    // face for use with h-refinement and AMR.
    double local_weight = 1.;
    std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, double,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
        neighbor_weights;
    for (const auto& kv : neighbor_vars) {
      local_weight -= neighbor_linear_weight;
      neighbor_weights[kv.first] = neighbor_linear_weight;
    }

    // Update `local_weights` and `neighbor_weights` to hold the unnormalized
    // nonlinear weights.
    local_weight = unnormalized_nonlinear_weight(
        local_weight, oscillation_indicator(local_polynomial, mesh));
    for (const auto& kv : neighbor_vars) {
      const auto& key = kv.first;
      const auto& neighbor_tensor_component =
          get<Tag>(kv.second)[tensor_storage_index];
      neighbor_weights[key] = unnormalized_nonlinear_weight(
          neighbor_weights[key],
          oscillation_indicator(neighbor_tensor_component, mesh));
    }

    // Update `local_weights` and `neighbor_weights` to hold the normalized
    // weights; these are the final weights of the WENO reconstruction.
    double normalization = local_weight;
    for (const auto& kv : neighbor_weights) {
      normalization += kv.second;
    }
    local_weight /= normalization;
    for (auto& kv : neighbor_weights) {
      kv.second /= normalization;
    }

    // Perform reconstruction, by superposition of local and neighbor
    // polynomials.
    const double mean = mean_value(local_polynomial, mesh);
    local_polynomial = mean + local_weight * (local_polynomial - mean);
    for (const auto& kv : neighbor_vars) {
      const auto& key = kv.first;
      const auto& neighbor_polynomial =
          get<Tag>(kv.second)[tensor_storage_index];
      const double neighbor_mean = mean_value(neighbor_polynomial, mesh);
      local_polynomial +=
          neighbor_weights.at(key) * (neighbor_polynomial - neighbor_mean);
    }
  }
}

}  // namespace Weno_detail
}  // namespace SlopeLimiters
