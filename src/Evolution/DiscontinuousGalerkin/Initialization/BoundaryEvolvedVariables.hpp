// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryVariables.hpp"
#include "Domain/BoundaryVariablesTag.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedVariables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Tags/LtsMode.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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

namespace evolution::dg::Initialization {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Allocates and initializes the system's boundary-evolved variables
/// (the `::Tags::BoundaryVariables` entry of the list-valued
/// `System::variables_tag`) on each opting external face.
///
/// Iterates the element's external boundaries and gives each face whose
/// boundary condition opts in (declares the `evolves_boundary_variables`
/// marker) a face-sized entry in the `BoundaryVariables` storage. Each
/// `Tags::BoundaryValue<Source>` is initialized by projecting its interior
/// `Source` from the volume variables to the face. Interior and non-opting
/// faces get no entry.
///
/// Runs after the domain and the initial data are set, and before
/// `Initialization::TimeStepperHistory`, which sizes the `::Tags::dt` twin of
/// the storage from the per-face storage initialized here.
///
/// \tparam Dim the spatial dimension
/// \tparam System the evolution system (supplies `variables_tag`)
/// \tparam DerivedBoundaryConditionsList the concrete boundary condition
/// types; it must contain every type that can appear on an external boundary
template <size_t Dim, typename System, typename DerivedBoundaryConditionsList>
struct BoundaryEvolvedVariables {
 private:
  static_assert(
      evolution::dg::system_has_boundary_variables_v<System>,
      "Boundary-evolved variables can only be initialized for a system whose "
      "variables_tag is a tmpl::list with a ::Tags::BoundaryVariables entry "
      "holding the boundary-evolved variables.");
  using boundary_variables_tag = evolution::dg::boundary_variables_tag<System>;
  using boundary_field_tags_list = typename boundary_variables_tag::tags_list;
  using volume_variables_tag = tmpl::front<typename System::variables_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<domain::Tags::ExternalBoundaryConditions<Dim>>;
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags = tmpl::list<boundary_variables_tag>;
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
    if (db::get<::Tags::LtsMode>(box) != LtsMode::Off) {
      ERROR(
          "Boundary-evolved variables are unverified with local time "
          "stepping.");
    }
    const auto& element = db::get<::domain::Tags::Element<Dim>>(box);
    const auto& mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    if (db::get<volume_variables_tag>(box).number_of_grid_points() !=
        mesh.number_of_grid_points()) {
      ERROR("The volume variables are not allocated to the mesh size ("
            << db::get<volume_variables_tag>(box).number_of_grid_points()
            << " variables grid points vs " << mesh.number_of_grid_points()
            << " mesh grid points). This action projects the volume "
               "variables onto the opting external faces, so it must run "
               "after the volume variables are allocated and set from the "
               "initial data.");
    }
    const auto& external_boundary_conditions =
        db::get<domain::Tags::ExternalBoundaryConditions<Dim>>(box).at(
            element.id().block_id());

    DirectionMap<Dim, size_t> points_per_direction{};
    for (const Direction<Dim>& direction : element.external_boundaries()) {
      call_with_dynamic_type<void, DerivedBoundaryConditionsList>(
          &*external_boundary_conditions.at(direction),
          [&direction, &mesh,
           &points_per_direction]<typename DerivedBoundaryCondition>(
              const DerivedBoundaryCondition* const /*boundary_condition*/) {
            if constexpr (evolution::dg::evolves_boundary_variables_v<
                              DerivedBoundaryCondition>) {
              points_per_direction.insert(
                  {direction, mesh.slice_away(direction.dimension())
                                  .number_of_grid_points()});
            }
          });
    }

    db::mutate<boundary_variables_tag>(
        [&mesh, &points_per_direction](
            const gsl::not_null<typename boundary_variables_tag::type*>
                boundary_vars,
            const typename volume_variables_tag::type& volume_variables) {
          boundary_vars->initialize(std::move(points_per_direction));
          for (auto& direction_and_face_values : boundary_vars->variables()) {
            const auto& direction = direction_and_face_values.first;
            auto& face_values = direction_and_face_values.second;
            tmpl::for_each<boundary_field_tags_list>(
                [&direction, &mesh, &volume_variables,
                 &face_values]<typename BoundaryTag>(
                    tmpl::type_<BoundaryTag> /*meta*/) {
                  using source_tag = typename BoundaryTag::tag;
                  auto& face_field = get<BoundaryTag>(face_values);
                  ::dg::project_tensor_to_boundary(
                      make_not_null(&face_field),
                      get<source_tag>(volume_variables), mesh, direction);
                });
          }
        },
        make_not_null(&box), db::get<volume_variables_tag>(box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::Initialization
