// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Elliptic/Initialization/BoundaryConditions.hpp"
#include "Elliptic/Initialization/DiscontinuousGalerkin.hpp"
#include "Elliptic/Initialization/Interface.hpp"
#include "Elliptic/Initialization/LinearSolver.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
template <size_t VolumeDim>
class ElementIndex;
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace tuples {
template <typename... Tags>
class TaggedTuple;  // IWYU pragma: keep
}  // namespace tuples
/// \endcond

namespace elliptic {
namespace dg {
namespace Actions {

/*!
 * \brief Initializes the DataBox of each element in the DgElementArray
 *
 * The following initializers are chained together (in this order):
 *
 * - `elliptic::Initialization::Interface`
 * - `elliptic::Initialization::BoundaryConditions`
 * - `elliptic::Initialization::LinearSolver`
 * - `elliptic::Initialization::DiscontinuousGalerkin`
 */
template <size_t Dim>
struct InitializeElement {
  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, ::Tags::InitialExtents<Dim>>> =
          nullptr>
  static auto apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementIndex<Dim>& array_index, const ActionList /*meta*/,
      const ParallelComponent* const parallel_component_meta) noexcept {
    const auto& initial_extents = db::get<::Tags::InitialExtents<Dim>>(box);

    using system = typename Metavariables::system;
    auto face_box =
        elliptic::Initialization::Interface<system>::initialize(std::move(box));
    auto boundary_conditions_box =
        elliptic::Initialization::BoundaryConditions<Metavariables>::initialize(
            std::move(face_box), cache);
    auto linear_solver_box =
        elliptic::Initialization::LinearSolver<Metavariables>::initialize(
            std::move(boundary_conditions_box), cache, array_index,
            parallel_component_meta);
    auto dg_box = elliptic::Initialization::DiscontinuousGalerkin<
        Metavariables>::initialize(std::move(linear_solver_box),
                                   initial_extents);
    return std::make_tuple(std::move(dg_box));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                DbTagsList, ::Tags::InitialExtents<Dim>>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box), true};
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
