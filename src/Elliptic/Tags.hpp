// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/IterationId.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"

namespace Elliptic {

/*!
 * \brief The \ref DataBoxGroup tags for elliptic solves
 */
namespace Tags {

/*!
 * \brief Holds an `Elliptic::IterationId` that identifies a step in the
 * elliptic solver algorithm
 */
template <typename... ComponentTags>
struct IterationId : db::SimpleTag {
  using type = Elliptic::IterationId<ComponentTags...>;
  static std::string name() noexcept { return "EllipticIterationId"; }
  template <typename Tag>
  using step_prefix =
      typename LinearSolver::Tags::IterationId::template step_prefix<Tag>;
};

/*!
 * \brief Computes the `Elliptic::Tags::IterationId` from the `ComponentTags`
 */
template <typename... ComponentTags>
struct IterationIdCompute : db::ComputeTag, IterationId<ComponentTags...> {
  using argument_tags = tmpl::list<ComponentTags...>;
  static Elliptic::IterationId<ComponentTags...> function(
      const db::item_type<ComponentTags>&... component_ids) noexcept {
    return {component_ids...};
  }
};

/*!
 * \brief Computes the `::Tags::Next<Elliptic::Tags::IterationId>` from the
 * `::Tags::Next<ComponentTags>`
 */
template <typename... ComponentTags>
struct NextIterationIdCompute : db::ComputeTag,
                                ::Tags::Next<IterationId<ComponentTags...>> {
  using argument_tags =
      db::wrap_tags_in<::Tags::Next, tmpl::list<ComponentTags...>>;
  static Elliptic::IterationId<ComponentTags...> function(
      const db::item_type<ComponentTags>&... next_component_ids) noexcept {
    return {next_component_ids...};
  }
};

}  // namespace Tags
}  // namespace Elliptic
