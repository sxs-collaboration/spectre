// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace dg {
namespace Actions {
/*!
 * \ingroup InitializationGroup
 * \brief Initialize items related to the basic structure of the element
 *
 * ConstGlobalCache:
 * - Uses:
 *   - `Tags::Domain<Dim, Frame::Inertial>`
 * DataBox:
 * - Uses:
 *   - `Tags::InitialExtents<Dim>`
 * - Adds:
 *   - `Tags::Mesh<Dim>`
 *   - `Tags::Element<Dim>`
 *   - `Tags::ElementMap<Dim, Frame::Inertial>`
 *   - `Tags::Coordinates<Dim, Frame::Logical>`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *   - `Tags::InverseJacobianCompute<
 *   Tags::ElementMap<Dim>, Tags::Coordinates<Dim, Frame::Logical>>`
 *   - `Tags::DetInvJacobianCompute<Dim, Frame::Logical, Frame::Inertial>`
 *   - `Tags::MinimumGridSpacing<Dim, Frame::Inertial>>`
 * - Removes: nothing
 * - Modifies: nothing
 */
template <size_t Dim>
struct InitializeDomain {
  using initialization_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>>;

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl::all<initialization_tags,
                         tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                    tmpl::pin<DataBox>>>::value> = nullptr>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags =
        db::AddSimpleTags<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                          domain::Tags::ElementMap<Dim>>;
    using compute_tags = tmpl::append<db::AddComputeTags<
        domain::Tags::LogicalCoordinates<Dim>,
        domain ::Tags::MappedCoordinates<
            domain::Tags::ElementMap<Dim>,
            domain ::Tags::Coordinates<Dim, Frame::Logical>>,
        domain ::Tags::InverseJacobianCompute<
            domain ::Tags::ElementMap<Dim>,
            domain::Tags::Coordinates<Dim, Frame::Logical>>,
        domain::Tags::DetInvJacobianCompute<Dim, Frame::Logical,
                                            Frame::Inertial>,
        domain::Tags::MinimumGridSpacing<Dim, Frame::Inertial>>>;

    const auto& initial_extents =
        db::get<domain::Tags::InitialExtents<Dim>>(box);
    const auto& initial_refinement =
        db::get<domain::Tags::InitialRefinementLevels<Dim>>(box);
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);

    const ElementId<Dim> element_id{array_index};
    const auto& my_block = domain.blocks()[element_id.block_id()];
    Mesh<Dim> mesh = domain::Initialization::create_initial_mesh(
        initial_extents, element_id);
    Element<Dim> element = domain::Initialization::create_initial_element(
        element_id, my_block, initial_refinement);
    if (my_block.is_time_dependent()) {
      ERROR(
          "The version of the InitializeDomain action being used is for "
          "elliptic systems which do not have any time-dependence but the "
          "domain creator has set up the domain to have time-dependence.");
    }
    ElementMap<Dim, Frame::Inertial> element_map{
        element_id, my_block.stationary_map().get_clone()};

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeDomain, simple_tags,
                                             compute_tags>(
            std::move(box), std::move(mesh), std::move(element),
            std::move(element_map)));
  }

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::all<initialization_tags,
                             tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                        tmpl::pin<DataBox>>>::value> = nullptr>
  static std::tuple<DataBox&&> apply(
      DataBox& /*box*/, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Dependencies not fulfilled. Did you forget to terminate the phase "
        "after removing options?");
  }
};
}  // namespace Actions
}  // namespace dg
