// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"          // IWYU pragma: keep
#include "Domain/ElementId.hpp"          // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {

// Compute the unnormalized nonlinear WENO weights. This is a fairly standard
// choice of weights; see e.g., Eq. 3.9 of Zhong2013 or Eq. 3.6 of Zhu2016.
inline double unnormalized_nonlinear_weight(
    const double linear_weight, const double oscillation_indicator) noexcept {
  return linear_weight / square(1.e-6 + oscillation_indicator);
}

}  // namespace

namespace Limiters {
namespace Weno_detail {

template <size_t VolumeDim>
void reconstruct_from_weighted_sum(
    const gsl::not_null<DataVector*> local_polynomial,
    const Mesh<VolumeDim>& mesh, const double neighbor_linear_weight,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_polynomials,
    const DerivativeWeight derivative_weight) noexcept {
#ifdef SPECTRE_DEBUG
  ASSERT(local_polynomial->size() > 0,
         "The local_polynomial values are missing - was the input correctly\n"
         "initialized before calling reconstruct_from_weighted_sum?");
  // Check inputs match requirements
  const double local_mean = mean_value(*local_polynomial, mesh);
  for (const auto& kv : neighbor_polynomials) {
    const auto& neighbor_polynomial = kv.second;
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
  for (const auto& kv : neighbor_polynomials) {
    local_weight -= neighbor_linear_weight;
    neighbor_weights[kv.first] = neighbor_linear_weight;
  }

  // Update `local_weights` and `neighbor_weights` to hold the unnormalized
  // nonlinear weights.
  local_weight = unnormalized_nonlinear_weight(
      local_weight,
      oscillation_indicator(*local_polynomial, mesh, derivative_weight));
  for (const auto& kv : neighbor_polynomials) {
    const auto& key = kv.first;
    const auto& neighbor_polynomial = kv.second;
    neighbor_weights[key] = unnormalized_nonlinear_weight(
        neighbor_weights[key],
        oscillation_indicator(neighbor_polynomial, mesh, derivative_weight));
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

  // Perform reconstruction by combining the local and neighbor polynomials.
  *local_polynomial *= local_weight;
  for (const auto& kv : neighbor_polynomials) {
    const auto& key = kv.first;
    const auto& neighbor_polynomial = kv.second;
    *local_polynomial += neighbor_weights.at(key) * neighbor_polynomial;
  }
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void reconstruct_from_weighted_sum(                                \
      const gsl::not_null<DataVector*>, const Mesh<DIM(data)>&, const double, \
      const std::unordered_map<                                               \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>, DataVector,  \
          boost::hash<                                                        \
              std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>&,       \
      const DerivativeWeight) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Weno_detail
}  // namespace Limiters
