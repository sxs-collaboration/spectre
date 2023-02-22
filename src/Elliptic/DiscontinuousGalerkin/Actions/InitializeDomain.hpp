// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Elliptic/DiscontinuousGalerkin/Initialization.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace elliptic::dg::Actions {
/*!
 * \ingroup InitializationGroup
 * \brief Initialize items related to the basic structure of the element
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Domain<Dim, Frame::Inertial>`
 *   - `domain::Tags::InitialExtents<Dim>`
 *   - `domain::Tags::InitialRefinementLevels<Dim>`
 * - Adds:
 *   - `domain::Tags::Mesh<Dim>`
 *   - `domain::Tags::Element<Dim>`
 *   - `domain::Tags::ElementMap<Dim, Frame::Inertial>`
 *   - `domain::Tags::Coordinates<Dim, Frame::ElementLogical>`
 *   - `domain::Tags::Coordinates<Dim, Frame::Inertial>`
 *   - `domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
 *      Frame::Inertial>`
 *   - `domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>`
 */
template <size_t Dim>
struct InitializeDomain {
 private:
  using InitializeGeometry = elliptic::dg::InitializeGeometry<Dim>;

 public:
  using simple_tags_from_options =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>>;
  using simple_tags = typename InitializeGeometry::return_tags;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<typename InitializeGeometry::return_tags,
                     typename InitializeGeometry::argument_tags>(
        InitializeGeometry{}, make_not_null(&box), element_id);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace elliptic::dg::Actions
