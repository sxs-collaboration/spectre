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
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
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

namespace Limiters {
namespace Weno_detail {

// Compute the unnormalized nonlinear WENO weights. This is a fairly standard
// choice of weights; see e.g., Eq. 3.9 of Zhong2013 or Eq. 3.6 of Zhu2016.
inline double unnormalized_nonlinear_weight(
    const double linear_weight, const double oscillation_indicator) noexcept {
  return linear_weight / square(1.e-6 + oscillation_indicator);
}

// Compute the WENO weighted reconstruction of several polynomials, see e.g.,
// Eq. 4.3 of Zhu2016. This is fairly standard, though different references can
// differ in their choice of oscillation/smoothness indicator.
//
// The neighbor tensors corresponding to `local_tensor` (i.e., the tensor in
// each `neighbor_vars` value that is identified by `Tag`), must have the same
// mean as the local tensor. This is checked with an ASSERT.
template <typename Tag, size_t VolumeDim, typename TagsList>
void reconstruct_from_weighted_sum(
    const gsl::not_null<db::item_type<Tag>*> local_tensor,
    const Mesh<VolumeDim>& mesh, const double neighbor_linear_weight,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        Variables<TagsList>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_vars,
    const DerivativeWeight derivative_weight) noexcept {
  for (size_t tensor_storage_index = 0;
       tensor_storage_index < local_tensor->size(); ++tensor_storage_index) {
    auto& local_polynomial = (*local_tensor)[tensor_storage_index];

#ifdef SPECTRE_DEBUG
    // Check inputs match requirements
    const double local_mean = mean_value(local_polynomial, mesh);
    for (const auto& kv : neighbor_vars) {
      const auto& neighbor_polynomial =
          get<Tag>(kv.second)[tensor_storage_index];
      const double neighbor_mean = mean_value(neighbor_polynomial, mesh);
      ASSERT(equal_within_roundoff(local_mean, neighbor_mean),
             "Invalid inputs to Weno_detail::reconstruct_from_weighted_sum:\n"
             "The neighbor polynomials should have the same mean as the local\n"
             "polynomial.");
    }
#endif  // ifdef SPECTRE_DEBUG

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
        local_weight,
        oscillation_indicator(local_polynomial, mesh, derivative_weight));
    for (const auto& kv : neighbor_vars) {
      const auto& key = kv.first;
      const auto& neighbor_tensor_component =
          get<Tag>(kv.second)[tensor_storage_index];
      neighbor_weights[key] = unnormalized_nonlinear_weight(
          neighbor_weights[key],
          oscillation_indicator(neighbor_tensor_component, mesh,
                                derivative_weight));
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
    local_polynomial *= local_weight;
    for (const auto& kv : neighbor_vars) {
      const auto& key = kv.first;
      const auto& neighbor_polynomial =
          get<Tag>(kv.second)[tensor_storage_index];
      local_polynomial += neighbor_weights.at(key) * neighbor_polynomial;
    }
  }
}

}  // namespace Weno_detail
}  // namespace Limiters
