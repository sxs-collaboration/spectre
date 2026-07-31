// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <typeinfo>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::BoundaryEvolvedFields {
namespace detail {
// A metavariables enables adaptive mesh refinement by declaring a nested `amr`
// type (which holds the `amr::projectors` list read by `amr::AdjustDomain`).
// Detect its presence to fail loud when the facility -- which stores per-face
// value/history maps with no AMR projector -- would otherwise have those maps
// silently emptied by an AMR event.
CREATE_HAS_TYPE_ALIAS(amr)
CREATE_HAS_TYPE_ALIAS_V(amr)
}  // namespace detail

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Allocates and initializes the per-face boundary-evolved
/// field storage on each opting external face.
///
/// Iterates the element's external boundaries, resolves each face's applied
/// boundary condition by `typeid` against `DerivedBoundaryConditionsList`, and,
/// for each boundary condition that opts in (declares
/// `boundary_evolved_variables`), allocates the face-sized per-face `Variables`
/// entries in the value, dt-stash, and history maps. Each
/// `Tags::BoundaryValue<Source>` is initialized by projecting its interior
/// `Source` from the volume `variables_tag` to the face. Interior and
/// non-opting faces get no entry.
///
/// Runs after the domain and initial data are set and after
/// `Initialization::Mortars`, before the self-start phase.
///
/// \tparam Dim the spatial dimension
/// \tparam System the evolution system (supplies `variables_tag`)
/// \tparam DerivedBoundaryConditionsList the concrete boundary condition types
template <size_t Dim, typename System, typename DerivedBoundaryConditionsList>
struct InitializeBoundaryEvolvedFields {
 private:
  using field_tags_list =
      boundary_evolved_field_tags<DerivedBoundaryConditionsList>;

  static_assert(
      boundary_evolved_fields_are_homogeneous_v<DerivedBoundaryConditionsList>,
      "The boundary-evolved-fields facility requires every boundary condition "
      "that opts in (declares `boundary_evolved_variables`) to declare the "
      "identical field list. Heterogeneous per-face field sets are not yet "
      "supported: a face would carry union components its own boundary "
      "condition never initializes, which are still recorded and "
      "time-integrated.");

 public:
  using values_tag = Tags::BoundaryEvolvedFieldsValues<Dim, field_tags_list>;
  using dt_stash_tag = Tags::BoundaryEvolvedFieldsDtStash<Dim, field_tags_list>;
  using history_tag = Tags::BoundaryEvolvedFieldsHistory<Dim, field_tags_list>;

  using const_global_cache_tags =
      tmpl::list<domain::Tags::ExternalBoundaryConditions<Dim>>;
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags = tmpl::list<values_tag, dt_stash_tag, history_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        tmpl::size<field_tags_list>::value == 0 or
            not detail::has_amr_v<Metavariables>,
        "Boundary-evolved fields do not yet support adaptive mesh refinement "
        "(AMR): it requires adding a boundary-history projector to the "
        "facility.");
    const auto& element = db::get<::domain::Tags::Element<Dim>>(box);
    const auto& mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const auto& volume_variables = db::get<typename System::variables_tag>(box);
    const auto& external_boundary_conditions =
        db::get<domain::Tags::ExternalBoundaryConditions<Dim>>(box).at(
            element.id().block_id());

    using volume_history_tags = ::Tags::get_all_history_tags<DbTagsList>;
    const size_t initial_order =
        db::get<tmpl::front<volume_history_tags>>(box).integration_order();

    typename values_tag::type values{};
    typename dt_stash_tag::type dt_stash{};
    typename history_tag::type histories{};

    for (const Direction<Dim>& direction : element.external_boundaries()) {
      const auto& boundary_condition =
          *external_boundary_conditions.at(direction);
      tmpl::for_each<DerivedBoundaryConditionsList>([&boundary_condition,
                                                     &direction, &mesh,
                                                     &volume_variables, &values,
                                                     &dt_stash, &histories,
                                                     &initial_order](
                                                        auto derived_bc_v) {
        using DerivedBoundaryCondition =
            tmpl::type_from<decltype(derived_bc_v)>;
        using bc_field_tags =
            boundary_evolved_variables_of<DerivedBoundaryCondition>;
        if constexpr (bc_opts_in_v<DerivedBoundaryCondition>) {
          if (typeid(boundary_condition) == typeid(DerivedBoundaryCondition)) {
            const size_t number_of_face_grid_points =
                mesh.slice_away(direction.dimension()).number_of_grid_points();
            Variables<field_tags_list> face_values{number_of_face_grid_points};
            tmpl::for_each<bc_field_tags>([&direction, &mesh, &volume_variables,
                                           &face_values](auto tag_v) {
              using boundary_tag = tmpl::type_from<decltype(tag_v)>;
              using source_tag = typename boundary_tag::tag;
              const auto& volume_field = get<source_tag>(volume_variables);
              auto& face_field = get<boundary_tag>(face_values);
              ::dg::project_tensor_to_boundary(make_not_null(&face_field),
                                               volume_field, mesh, direction);
            });
            dt_stash.insert(
                {direction, typename dt_stash_tag::type::mapped_type{
                                number_of_face_grid_points}});
            histories.insert(
                {direction,
                 typename history_tag::type::mapped_type{initial_order}});
            values.insert({direction, std::move(face_values)});
          }
        }
      });
    }

    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(values), std::move(dt_stash),
        std::move(histories));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::BoundaryEvolvedFields
