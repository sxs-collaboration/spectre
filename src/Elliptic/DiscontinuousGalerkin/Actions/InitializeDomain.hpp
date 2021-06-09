// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Initialization.hpp"
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
 *   - `domain::Tags::Coordinates<Dim, Frame::Logical>`
 *   - `domain::Tags::Coordinates<Dim, Frame::Inertial>`
 *   - `domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>`
 *   - `domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <size_t Dim>
struct InitializeDomain {
 private:
  using InitializeGeometry = elliptic::dg::InitializeGeometry<Dim>;

 public:
  using initialization_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>>;
  using simple_tags = typename InitializeGeometry::return_tags;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate_apply<typename InitializeGeometry::return_tags,
                     typename InitializeGeometry::argument_tags>(
        InitializeGeometry{}, make_not_null(&box), element_id);
    return {std::move(box)};
  }
};
}  // namespace elliptic::dg::Actions
