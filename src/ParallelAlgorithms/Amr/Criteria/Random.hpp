// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Criteria {
/*!
 * \brief Randomly h-refine (or coarsen) an Element in each dimension.
 *
 * \details Let \f$f\f$ be `ChangeRefinementFraction`, \f$L_{max}\f$ be
 * `MaximumRefinementLevel`, and \f$L_d\f$ be the current refinement level
 * of an Element in a particular dimension.  In each dimension, a random
 * number \f$r_d \in [0, 1]\f$ is generated.  If \f$r_d > f\f$ the refinement
 * flag is set to amr::Flags::DoNothing.  If \f$r_d < f L_d / L_{max}\f$
 * the refinement flag is set to amr::Flags::Join.  Otherwise the
 * refinement flag is set to amr::Flag::Split.
 *
 * \note This criterion is primarily useful for testing the mechanics of
 * h-refinement
 */
class Random : public Criterion {
 public:
  /// The fraction of the time random refinement does changes the grid
  struct ChangeRefinementFraction {
    using type = double;
    static constexpr Options::String help = {
        "The fraction of the time that random refinement will change the "
        "grid."};
    static double lower_bound() { return 0.0; }
    static double upper_bound() { return 1.0; }
  };

  /// The maximum allowed refinement level
  struct MaximumRefinementLevel {
    using type = size_t;
    static constexpr Options::String help = {
        "The maximum allowed refinement level."};
    static size_t upper_bound() { return ElementId<3>::max_refinement_level; }
  };

  using options = tmpl::list<ChangeRefinementFraction, MaximumRefinementLevel>;

  static constexpr Options::String help = {
      "Randomly h-refine (or coarsen) the grid"};

  Random() = default;

  Random(const double do_something_fraction,
         const size_t maximum_refinement_level);

  /// \cond
  explicit Random(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Random);  // NOLINT
  /// \endcond

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags = tmpl::list<>;

  template <typename Metavariables>
  auto operator()(Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ElementId<Metavariables::volume_dim>& element_id) const;

  void pup(PUP::er& p) override;

 private:
  amr::Flag random_flag(size_t current_refinement_level) const;

  double do_something_fraction_{0.0};
  size_t maximum_refinement_level_{0};
};

template <typename Metavariables>
auto Random::operator()(
    Parallel::GlobalCache<Metavariables>& /*cache*/,
    const ElementId<Metavariables::volume_dim>& element_id) const {
  constexpr size_t volume_dim = Metavariables::volume_dim;
  auto result = make_array<volume_dim>(amr::Flag::Undefined);
  for (size_t d = 0; d < volume_dim; ++d) {
    result[d] = random_flag(element_id.segment_ids()[d].refinement_level());
  }
  return result;
}
}  // namespace amr::Criteria
