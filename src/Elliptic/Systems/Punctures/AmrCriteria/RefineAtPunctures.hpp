// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Punctures::AmrCriteria {

/*!
 * \brief h-refine (split) elements containing a puncture, and p-refine
 * everywhere else.
 *
 * This refinement scheme is expected to yield exponential convergence, despite
 * the presence of the C^2 punctures.
 */
class RefineAtPunctures : public amr::Criterion {
 public:
  using options = tmpl::list<>;

  static constexpr Options::String help = {
      "h-refine (split) elements containing a puncture, and p-refine "
      "everywhere else."};

  RefineAtPunctures() = default;

  /// \cond
  explicit RefineAtPunctures(CkMigrateMessage* msg) : Criterion(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(RefineAtPunctures);  // NOLINT
  /// \endcond

  using argument_tags = tmpl::list<
      elliptic::Tags::Background<elliptic::analytic_data::Background>,
      domain::Tags::Domain<3>>;
  using compute_tags_for_observation_box = tmpl::list<>;

  template <typename Metavariables>
  std::array<amr::Flag, 3> operator()(
      const elliptic::analytic_data::Background& background,
      const Domain<3>& domain, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<3>& element_id) const {
    return impl(background, domain, element_id);
  }

 private:
  static std::array<amr::Flag, 3> impl(
      const elliptic::analytic_data::Background& background,
      const Domain<3>& domain, const ElementId<3>& element_id);
};

}  // namespace Punctures::AmrCriteria
