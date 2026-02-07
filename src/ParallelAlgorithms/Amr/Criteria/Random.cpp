// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/Random.hpp"

#include <cstddef>
#include <random>
#include <unordered_map>

#include "Domain/Amr/Flag.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace amr::Criteria {
template <Type CriteriaType>
Random<CriteriaType>::Random(
    std::unordered_map<amr::Flag, size_t> probability_weights)
    : probability_weights_(std::move(probability_weights)) {
  if constexpr (CriteriaType == Type::h) {
    if (probability_weights_.contains(amr::Flag::IncreaseResolution) or
        probability_weights_.contains(amr::Flag::DecreaseResolution)) {
      ERROR("Cannot use p-refinement flag in ProbabilityWeights "
            << probability_weights_);
    }
  } else {
    if (probability_weights_.contains(amr::Flag::Split) or
        probability_weights_.contains(amr::Flag::Join)) {
      ERROR("Cannot use h-refinement flag in ProbabilityWeights "
            << probability_weights_);
    }
  }
}

// NOLINTNEXTLINE(google-runtime-references)
template <Type CriteriaType>
void Random<CriteriaType>::pup(PUP::er& p) {
  Criterion::pup(p);
  p | probability_weights_;
}

namespace detail {
amr::Flag random_flag(
    const std::unordered_map<amr::Flag, size_t>& probability_weights) {
  const size_t total_weight =
      alg::accumulate(probability_weights, 0_st,
                      [](const size_t total, const auto& flag_and_weight) {
                        return total + flag_and_weight.second;
                      });
  if (total_weight == 0) {
    return amr::Flag::DoNothing;
  }
  if (probability_weights.size() == 1) {
    return probability_weights.begin()->first;
  }

  static std::random_device r;
  static const auto seed = r();
  static std::mt19937 generator(seed);
  std::uniform_int_distribution<size_t> distribution{0, total_weight - 1};

  const size_t random_number = distribution(generator);
  size_t cumulative_weight = 0;
  for (const auto& [flag, probability_weight] : probability_weights) {
    cumulative_weight += probability_weight;
    if (random_number < cumulative_weight) {
      return flag;
    }
  }
  ERROR("Should never reach here");
}
}  // namespace detail

#if defined(SPECTRE_USE_CHARM)
template <Type CriteriaType>
PUP::able::PUP_ID Random<CriteriaType>::my_PUP_ID = 0;  // NOLINT
#endif                                                  // SPECTRE_USE_CHARM

template class Random<Type::h>;
template class Random<Type::p>;
}  // namespace amr::Criteria
