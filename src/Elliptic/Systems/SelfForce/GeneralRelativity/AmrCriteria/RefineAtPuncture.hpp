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
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::AmrCriteria {

/*!
 * \brief h-refine (split) elements containing a puncture
 *
 * This refinement scheme is expected to yield exponential convergence, despite
 * the presence of the C^2 punctures.
 */
class RefineAtPuncture : public amr::Criterion {
 public:
  using options = tmpl::list<>;

  static constexpr Options::String help = {
      "h-refine (split) elements containing a puncture."};

  RefineAtPuncture() = default;

  /// \cond
  explicit RefineAtPuncture(CkMigrateMessage* msg) : Criterion(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(RefineAtPuncture);  // NOLINT
  /// \endcond

  amr::Criteria::Type type() override { return amr::Criteria::Type::h; }

  std::string observation_name() override { return "RefineAtPuncture"; }

  using argument_tags = tmpl::list<
      elliptic::Tags::Background<elliptic::analytic_data::Background>,
      domain::Tags::Domain<2>>;
  using compute_tags_for_observation_box = tmpl::list<>;

  template <typename Metavariables>
  std::array<amr::Flag, 2> operator()(
      const elliptic::analytic_data::Background& background,
      const Domain<2>& domain, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<2>& element_id) const {
    return impl(background, domain, element_id);
  }

 private:
  static std::array<amr::Flag, 2> impl(
      const elliptic::analytic_data::Background& background,
      const Domain<2>& domain, const ElementId<2>& element_id);
};

}  // namespace GrSelfForce::AmrCriteria
