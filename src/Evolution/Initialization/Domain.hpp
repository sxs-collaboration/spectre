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
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Helpers.hpp"
#include "Evolution/Initialization/MergeIntoDataBox.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t VolumeDim>
class ElementIndex;
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace Initialization {
namespace Actions {
/// \brief Initialize items related to the basic structure of the Domain
///
/// DataBox changes:
/// - Adds:
///   * `Tags::Mesh<Dim>`
///   * `Tags::Element<Dim>`
///   * `Tags::ElementMap<Dim>`
///   * `Tags::LogicalCoordinates<Dim>`
///   * `Tags::MappedCoordinates`
///   * `Tags::InverseJacobian`
///   * `Tags::MinimumGridSpacing<Dim, Frame::Inertial>>`
/// - Removes: nothing
/// - Modifies: nothing
template <size_t Dim>
struct Domain {
  using simple_tags = db::AddSimpleTags<::Tags::Mesh<Dim>, ::Tags::Element<Dim>,
                                        ::Tags::ElementMap<Dim>>;

  using compute_tags = db::AddComputeTags<
      ::Tags::LogicalCoordinates<Dim>,
      ::Tags::MappedCoordinates<::Tags::ElementMap<Dim>,
                                ::Tags::LogicalCoordinates<Dim>>,
      ::Tags::InverseJacobian<::Tags::ElementMap<Dim>,
                              ::Tags::LogicalCoordinates<Dim>>,
      ::Tags::MinimumGridSpacing<Dim, Frame::Inertial>>;

  using initialization_option_tags =
      tmpl::list<Tags::InitialExtents<Dim>, Tags::Domain<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                typename db::DataBox<DbTagsList>::simple_item_tags,
                Tags::InitialExtents<Dim>>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& array_index, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& initial_extents = db::get<Tags::InitialExtents<Dim>>(box);
    const ::Domain<Dim, Frame::Inertial>& domain =
        db::get<Tags::Domain<Dim>>(box);
    const ElementId<Dim> element_id{array_index};
    const auto& my_block = domain.blocks()[element_id.block_id()];
    Mesh<Dim> mesh = element_mesh(initial_extents, element_id);
    Element<Dim> element = create_initial_element(element_id, my_block);
    ElementMap<Dim, Frame::Inertial> map{element_id,
                                         my_block.coordinate_map().get_clone()};

    return std::make_tuple(
        merge_into_databox<Domain, simple_tags, compute_tags>(
            std::move(box), std::move(mesh), std::move(element),
            std::move(map)));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                typename db::DataBox<DbTagsList>::simple_item_tags,
                Tags::InitialExtents<Dim>>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Could not find dependency 'Initialization::Tags::InitialExtents' in "
        "DataBox.");
  }
};
}  // namespace Actions
}  // namespace Initialization
