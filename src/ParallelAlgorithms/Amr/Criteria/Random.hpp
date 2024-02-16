// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <pup.h>
#include <unordered_map>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Criteria {
namespace detail {
amr::Flag random_flag(
    const std::unordered_map<amr::Flag, size_t>& probability_weights);
}  // namespace detail

/*!
 * \brief Randomly refine (or coarsen) an Element in each dimension.
 *
 * You can specify a probability for each possible `amr::Flag`. It is evaluated
 * in each dimension separately. Details:
 *
 * - Probabilities are specified as integer weights. The probability for an
 *   `amr::Flag` is its weight over the sum of all weights.
 * - Flags with weight zero do not need to be specified.
 * - If all weights are zero, `amr::Flag::DoNothing` is always chosen.
 */
class Random : public Criterion {
 public:
  struct ProbabilityWeights {
    using type = std::unordered_map<amr::Flag, size_t>;
    static constexpr Options::String help = {
        "Possible refinement types and their probability, specified as integer "
        "weights. The probability for a refinement type is its weight over the "
        "sum of all weights. For example, set 'Split: 1' and 'DoNothing: 4' to "
        "split each element with 20% probability. The refinement is evaluated "
        "in each dimension separately."};
  };

  /// The maximum allowed refinement level.
  /// Can be deleted once the max refinement level is enforced globally as an
  /// AMR policy.
  struct MaximumRefinementLevel {
    using type = size_t;
    static constexpr Options::String help = {
        "The maximum allowed refinement level."};
    static size_t upper_bound() { return ElementId<3>::max_refinement_level; }
  };

  using options = tmpl::list<ProbabilityWeights, MaximumRefinementLevel>;

  static constexpr Options::String help = {
      "Randomly refine (or coarsen) the grid"};

  Random() = default;

  explicit Random(
      std::unordered_map<amr::Flag, size_t> probability_weights,
      size_t maximum_refinement_level = std::numeric_limits<size_t>::max());

  /// \cond
  explicit Random(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Random);  // NOLINT
  /// \endcond

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags = tmpl::list<>;

  template <size_t Dim, typename Metavariables>
  auto operator()(Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p) override;

 private:
  std::unordered_map<amr::Flag, size_t> probability_weights_{};
  size_t maximum_refinement_level_ = std::numeric_limits<size_t>::max();
};

template <size_t Dim, typename Metavariables>
auto Random::operator()(Parallel::GlobalCache<Metavariables>& /*cache*/,
                        const ElementId<Dim>& element_id) const {
  auto result = make_array<Dim>(amr::Flag::Undefined);
  for (size_t d = 0; d < Dim; ++d) {
    result[d] = detail::random_flag(probability_weights_);
    // Enforce max refinement level. Can be deleted once it's enforced globally.
    if (result[d] == amr::Flag::Split and
        element_id.segment_ids()[d].refinement_level() >=
            maximum_refinement_level_) {
      result[d] = amr::Flag::DoNothing;
    }
  }
  return result;
}
}  // namespace amr::Criteria
