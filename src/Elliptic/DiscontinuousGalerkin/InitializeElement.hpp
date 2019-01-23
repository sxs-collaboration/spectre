// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Domain.hpp"
#include "Elliptic/Initialization/BoundaryConditions.hpp"
#include "Elliptic/Initialization/Derivatives.hpp"
#include "Elliptic/Initialization/DiscontinuousGalerkin.hpp"
#include "Elliptic/Initialization/Domain.hpp"
#include "Elliptic/Initialization/Interface.hpp"
#include "Elliptic/Initialization/LinearSolver.hpp"
#include "Elliptic/Initialization/Source.hpp"
#include "Elliptic/Initialization/System.hpp"
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

namespace Elliptic {
namespace dg {
namespace Actions {

/*!
 * \brief Initializes the DataBox of each element in the DgElementArray
 *
 * The following initializers are chained together (in this order):
 *
 * - `Elliptic::Initialization::Domain`
 * - `Elliptic::Initialization::System`
 * - `Elliptic::Initialization::Source`
 * - `Elliptic::Initialization::Derivatives`
 * - `Elliptic::Initialization::Interface`
 * - `Elliptic::Initialization::BoundaryConditions`
 * - `Elliptic::Initialization::LinearSolver`
 * - `Elliptic::Initialization::DiscontinuousGalerkin`
 */
template <size_t Dim>
struct InitializeElement {
  template <class Metavariables>
  using return_tag_list = tmpl::append<
      // Simple tags
      typename Elliptic::Initialization::Domain<Dim>::simple_tags,
      typename Elliptic::Initialization::System<
          typename Metavariables::system>::simple_tags,
      typename Elliptic::Initialization::Source<Metavariables>::simple_tags,
      typename Elliptic::Initialization::Derivatives<
          typename Metavariables::system>::simple_tags,
      typename Elliptic::Initialization::Interface<
          typename Metavariables::system>::simple_tags,
      typename Elliptic::Initialization::BoundaryConditions<
          Metavariables>::simple_tags,
      typename Elliptic::Initialization::LinearSolver<
          Metavariables>::simple_tags,
      typename Elliptic::Initialization::DiscontinuousGalerkin<
          Metavariables>::simple_tags,
      // Compute tags
      typename Elliptic::Initialization::Domain<Dim>::compute_tags,
      typename Elliptic::Initialization::System<
          typename Metavariables::system>::compute_tags,
      typename Elliptic::Initialization::Source<Metavariables>::compute_tags,
      typename Elliptic::Initialization::Derivatives<
          typename Metavariables::system>::compute_tags,
      typename Elliptic::Initialization::Interface<
          typename Metavariables::system>::compute_tags,
      typename Elliptic::Initialization::BoundaryConditions<
          Metavariables>::compute_tags,
      typename Elliptic::Initialization::LinearSolver<
          Metavariables>::compute_tags,
      typename Elliptic::Initialization::DiscontinuousGalerkin<
          Metavariables>::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const parallel_component_meta,
                    std::vector<std::array<size_t, Dim>> initial_extents,
                    Domain<Dim, Frame::Inertial> domain) noexcept {
    using system = typename Metavariables::system;
    auto domain_box = Elliptic::Initialization::Domain<Dim>::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    auto system_box = Elliptic::Initialization::System<system>::initialize(
        std::move(domain_box));
    auto source_box =
        Elliptic::Initialization::Source<Metavariables>::initialize(
            std::move(system_box), cache);
    auto deriv_box = Elliptic::Initialization::Derivatives<
        typename Metavariables::system>::initialize(std::move(source_box));
    auto face_box = Elliptic::Initialization::Interface<system>::initialize(
        std::move(deriv_box));
    auto boundary_conditions_box =
        Elliptic::Initialization::BoundaryConditions<Metavariables>::initialize(
            std::move(face_box), cache);
    auto linear_solver_box =
        Elliptic::Initialization::LinearSolver<Metavariables>::initialize(
            std::move(boundary_conditions_box), cache, array_index,
            parallel_component_meta);
    auto dg_box = Elliptic::Initialization::DiscontinuousGalerkin<
        Metavariables>::initialize(std::move(linear_solver_box),
                                   initial_extents);
    return std::make_tuple(std::move(dg_box));
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace Elliptic
