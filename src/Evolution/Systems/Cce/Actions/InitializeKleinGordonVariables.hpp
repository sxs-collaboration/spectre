// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"

namespace Cce::Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Initialize the data storage for the scalar field in the
 * `KleinGordonCharacteristicExtract` component, which is the singleton that
 * handles the main evolution system for Klein-Gordon CCE computations.
 *
 * \details Sets up the \ref DataBoxGroup to be ready to take data from the
 * worldtube component, calculate initial data, and start the hypersurface
 * computations.
 *
 * \ref DataBoxGroup changes:
 * - Modifies: nothing
 * - Adds:
 *  - `Tags::Variables<metavariables::klein_gordon_boundary_communication_tags>`
 *  - `Tags::Variables<metavariables::klein_gordon_gauge_boundary_tags>`
 *  - `Tags::Variables<metavariables::klein_gordon_scri_tags>`
 *  - `Tags::Variables<metavariables::klein_gordon_pre_swsh_derivative_tags>`
 *  - `Tags::Variables<metavariables::klein_gordon_swsh_derivative_tags>`
 *  - `Tags::Variables<metavariables::klein_gordon_transform_buffer_tags>`
 *  - `Tags::Variables<metavariables::klein_gordon_source_tags>`
 *  - `Tags::Variables<metavariables::klein_gordon_cce_integrand_tags>`
 * - Removes: nothing
 */
template <typename Metavariables>
struct InitializeKleinGordonVariables {
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;

  using klein_gordon_boundary_communication_tags = ::Tags::Variables<
      typename Metavariables::klein_gordon_boundary_communication_tags>;
  using klein_gordon_gauge_boundary_tags = ::Tags::Variables<
      typename Metavariables::klein_gordon_gauge_boundary_tags>;
  using klein_gordon_scri_tags =
      ::Tags::Variables<typename Metavariables::klein_gordon_scri_tags>;
  using klein_gordon_pre_swsh_derivatives_variables_tag = ::Tags::Variables<
      typename Metavariables::klein_gordon_pre_swsh_derivative_tags>;
  using klein_gordon_swsh_derivatives_variables_tag = ::Tags::Variables<
      typename Metavariables::klein_gordon_swsh_derivative_tags>;
  using klein_gordon_transform_buffer_variables_tag = ::Tags::Variables<
      typename Metavariables::klein_gordon_transform_buffer_tags>;
  using klein_gordon_source_variables_tag =
      ::Tags::Variables<typename Metavariables::klein_gordon_source_tags>;
  using klein_gordon_integrand_variables_tag = ::Tags::Variables<
      typename Metavariables::klein_gordon_cce_integrand_tags>;
  using klein_gordon_constraint_variables_tag =
      ::Tags::Variables<typename Metavariables::klein_gordon_constraint_tags>;

  using simple_tags =
      tmpl::list<klein_gordon_boundary_communication_tags,
                 klein_gordon_gauge_boundary_tags, klein_gordon_scri_tags,
                 klein_gordon_pre_swsh_derivatives_variables_tag,
                 klein_gordon_swsh_derivatives_variables_tag,
                 klein_gordon_transform_buffer_variables_tag,
                 klein_gordon_source_variables_tag,
                 klein_gordon_integrand_variables_tag,
                 klein_gordon_constraint_variables_tag>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t l_max = db::get<Spectral::Swsh::Tags::LMaxBase>(box);
    const size_t number_of_radial_points =
        db::get<Spectral::Swsh::Tags::NumberOfRadialPointsBase>(box);
    const size_t boundary_size =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    const size_t volume_size = boundary_size * number_of_radial_points;
    const size_t transform_buffer_size =
        number_of_radial_points *
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);

    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box),
        typename klein_gordon_boundary_communication_tags::type{boundary_size},
        typename klein_gordon_gauge_boundary_tags::type{boundary_size},
        typename klein_gordon_scri_tags::type{boundary_size},
        typename klein_gordon_pre_swsh_derivatives_variables_tag::type{
            volume_size},
        typename klein_gordon_swsh_derivatives_variables_tag::type{volume_size},
        typename klein_gordon_transform_buffer_variables_tag::type{
            transform_buffer_size},
        typename klein_gordon_source_variables_tag::type{volume_size},
        typename klein_gordon_integrand_variables_tag::type{volume_size},
        typename klein_gordon_constraint_variables_tag::type{boundary_size});
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Cce::Actions
