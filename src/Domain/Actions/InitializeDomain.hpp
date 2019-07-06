// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/CreateInitialMesh.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t VolumeDim>
class ElementIndex;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain {
namespace Actions {

/*!
 * \ingroup InitializationGroup
 * \brief Initialize items related to the basic structure of the element
 *
 * DataBox:
 * - Uses:
 *   - `Tags::Domain<Dim, Frame::Inertial>`
 *   - `Tags::InitialExtents<Dim>`
 * - Adds:
 *   - `Tags::Mesh<Dim>`
 *   - `Tags::Element<Dim>`
 *   - `Tags::ElementMap<Dim, Frame::Inertial>`
 *   - `Tags::Coordinates<Dim, Frame::Logical>`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *   - `Tags::InverseJacobian<
 *   Tags::ElementMap<Dim>, Tags::Coordinates<Dim, Frame::Logical>>`
 *   - `Tags::MinimumGridSpacing<Dim, Frame::Inertial>>`
 */
template <size_t Dim, Initialization::MergePolicy MergePolicy =
                          Initialization::MergePolicy::Error>
struct InitializeDomain {
  using initialization_option_tags =
      tmpl::list<Tags::InitialExtents<Dim>, Tags::Domain<Dim, Frame::Inertial>>;

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<db::tag_is_retrievable_v<Tags::InitialExtents<Dim>, DataBox> and
               db::tag_is_retrievable_v<Tags::Domain<Dim, Frame::Inertial>,
                                        DataBox>> = nullptr>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags = db::AddSimpleTags<Tags::Mesh<Dim>, Tags::Element<Dim>,
                                          Tags::ElementMap<Dim>>;
    using compute_tags = tmpl::append<db::AddComputeTags<
        Tags::LogicalCoordinates<Dim>,
        Tags::MappedCoordinates<Tags::ElementMap<Dim>,
                                Tags::Coordinates<Dim, Frame::Logical>>,
        Tags::InverseJacobian<Tags::ElementMap<Dim>,
                              Tags::Coordinates<Dim, Frame::Logical>>,
        Tags::MinimumGridSpacing<Dim, Frame::Inertial>>>;

    const auto& initial_extents = db::get<Tags::InitialExtents<Dim>>(box);
    const auto& domain = db::get<Tags::Domain<Dim, Frame::Inertial>>(box);

    const ElementId<Dim> element_id{array_index};
    const auto& my_block = domain.blocks()[element_id.block_id()];
    Mesh<Dim> mesh = create_initial_mesh(initial_extents, element_id);
    Element<Dim> element = create_initial_element(element_id, my_block);
    ElementMap<Dim, Frame::Inertial> element_map{
        element_id, my_block.coordinate_map().get_clone()};

    return std::make_tuple(
        Initialization::merge_into_databox<InitializeDomain, simple_tags,
                                           compute_tags, MergePolicy>(
            std::move(box), std::move(mesh), std::move(element),
            std::move(element_map)));
  }

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<
          not db::tag_is_retrievable_v<Tags::InitialExtents<Dim>, DataBox> or
          not db::tag_is_retrievable_v<Tags::Domain<Dim, Frame::Inertial>,
                                       DataBox>> = nullptr>
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
}  // namespace domain
