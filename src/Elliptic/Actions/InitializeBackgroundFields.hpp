// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic::Actions {

/*!
 * \brief Initialize the variable-independent background fields for an elliptic
 * solve.
 *
 * Examples for background fields would be a background metric, associated
 * curvature quantities, or matter sources such as a mass-density in the XCTS
 * equations.
 *
 * This action retrieves the `System::background_fields` from the
 * `BackgroundTag`. It invokes the `variables` function with the inertial
 * coordinates, the element's `Mesh` and the element's inverse Jacobian. These
 * arguments allow computing numeric derivatives, if necessary.
 *
 * Uses:
 * - System:
 *   - `background_fields`
 * - DataBox:
 *   - `BackgroundTag`
 *   - `domain::Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `::Tags::Variables<background_fields>`
 */
template <typename System, typename BackgroundTag>
struct InitializeBackgroundFields {
  static_assert(
      not std::is_same_v<typename System::background_fields, tmpl::list<>>,
      "The system has no background fields. Don't add the "
      "'InitializeBackgroundFields' action to the action list.");

 private:
  using background_fields_tag =
      ::Tags::Variables<typename System::background_fields>;

 public:
  using simple_tags = tmpl::list<background_fields_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& background = db::get<BackgroundTag>(box);
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& mesh = get<domain::Tags::Mesh<Dim>>(box);
    const auto& inv_jacobian =
        get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                          Frame::Inertial>>(box);
    auto background_fields = variables_from_tagged_tuple(
        background.variables(inertial_coords, mesh, inv_jacobian,
                             typename background_fields_tag::tags_list{}));
    ::Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(background_fields));
    return {std::move(box)};
  }
};

}  // namespace elliptic::Actions
