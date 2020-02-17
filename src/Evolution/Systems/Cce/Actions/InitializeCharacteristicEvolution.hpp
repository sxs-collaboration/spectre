// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Systems/Cce/InitializeCce.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Parallel/Info.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
/// \brief The set of actions for use in the CCE evolution system
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Initializes the `CharacteristicEvolution` component, which is the
 * singleton that handles the main evolution system for CCE computations.
 *
 * \details Sets up the \ref DataBoxGroup to be ready to take data from the
 * worldtube component and calculate initial data.
 *
 * \ref DataBoxGroup changes:
 * - Modifies: nothing
 * - Adds:
 *  - `Cce::Tags::BoundaryTime`
 *  - `Tags::TimeStepId`
 *  - `Tags::Next<Tags::TimeStepId>`
 *  - `Tags::TimeStep`
 *  -
 * ```
 * Tags::HistoryEvolvedVariables<
 * metavariables::evolved_coordinates_variables_tag,
 * db::add_tag_prefix<Tags::dt,
 * metavariables::evolved_coordinates_variables_tag>>
 * ```
 *  -
 * ```
 * Tags::HistoryEvolvedVariables<
 *::Tags::Variables<metavariables::evolved_swsh_tag>,
 * ::Tags::Variables<metavariables::evolved_swsh_dt_tag>>
 * ```
 *  - `metavariables::evolved_coordinates_variables_tag`
 *  -
 * ```
 * db::add_tag_prefix<Tags::dt,
 * metavariables::evolved_coordinates_variables_tag>
 * ```
 *  - `Tags::Variables<metavariables::cce_angular_coordinate_tags>`
 *  - `Tags::Variables<metavariables::cce_scri_tags>`
 *  -
 * ```
 * Tags::Variables<tmpl::append<
 * metavariables::cce_integrand_tags,
 * metavariables::cce_integration_independent_tags,
 * metavariables::cce_temporary_equations_tags>>
 * ```
 *  - `Tags::Variables<metavariables::cce_pre_swsh_derivatives_tags>`
 *  - `Tags::Variables<metavariables::cce_transform_buffer_tags>`
 *  - `Tags::Variables<metavariables::cce_swsh_derivative_tags>`
 *  - `Spectral::Swsh::Tags::NumberOfRadialPoints`
 *  - `Tags::EndTime`
 *  - `Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>`
 * - Removes: nothing
 */
struct InitializeCharacteristicEvolution {
  using initialization_tags =
      tmpl::list<InitializationTags::StartTime, InitializationTags::EndTime,
                 InitializationTags::TargetStepSize>;
  using const_global_cache_tags =
      tmpl::list<::Tags::TimeStepper<TimeStepper>, Spectral::Swsh::Tags::LMax,
                 Spectral::Swsh::Tags::NumberOfRadialPoints>;

  template <typename Metavariables>
  struct EvolutionTags {
    using coordinate_variables_tag =
        typename Metavariables::evolved_coordinates_variables_tag;
    using dt_coordinate_variables_tag =
        db::add_tag_prefix<::Tags::dt, coordinate_variables_tag>;
    using evolution_simple_tags = db::AddSimpleTags<
        ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
        ::Tags::HistoryEvolvedVariables<coordinate_variables_tag>,
        ::Tags::HistoryEvolvedVariables<
            ::Tags::Variables<
                tmpl::list<typename Metavariables::evolved_swsh_tag>>>>;
    using evolution_compute_tags =
        db::AddComputeTags<::Tags::SubstepTimeCompute>;

    template <typename TagList>
    static auto initialize(
        db::DataBox<TagList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& /*cache*/) noexcept {
      const double initial_time_value =
          db::get<InitializationTags::StartTime>(box);
      const double step_size = db::get<InitializationTags::TargetStepSize>(box);

      const Slab single_step_slab{initial_time_value,
                                  initial_time_value + step_size};
      const Time initial_time = single_step_slab.start();
      const TimeDelta fixed_time_step =
          TimeDelta{single_step_slab, Rational{1, 1}};
      TimeStepId initial_time_id{true, 0, initial_time};
      const auto& time_stepper = db::get<::Tags::TimeStepper<TimeStepper>>(box);
      TimeStepId second_time_id =
          time_stepper.next_time_id(initial_time_id, fixed_time_step);

      db::item_type<::Tags::HistoryEvolvedVariables<coordinate_variables_tag>>
          coordinate_history;

      db::item_type<::Tags::HistoryEvolvedVariables<
          ::Tags::Variables<
              tmpl::list<typename Metavariables::evolved_swsh_tag>>>>
          swsh_history;

      return Initialization::merge_into_databox<
          InitializeCharacteristicEvolution, evolution_simple_tags,
          evolution_compute_tags, Initialization::MergePolicy::Overwrite>(
          std::move(box), std::move(initial_time_id),  // NOLINT
          std::move(second_time_id), fixed_time_step,  // NOLINT
          std::move(coordinate_history), std::move(swsh_history));
    }
  };

  template <typename Metavariables>
  struct CharacteristicTags {
    using boundary_value_variables_tag = ::Tags::Variables<
        tmpl::append<typename Metavariables::cce_boundary_communication_tags,
                     typename Metavariables::cce_gauge_boundary_tags>>;

    using scri_variables_tag =
        ::Tags::Variables<typename Metavariables::cce_scri_tags>;
    using volume_variables_tag = ::Tags::Variables<
        tmpl::append<typename Metavariables::cce_integrand_tags,
                     typename Metavariables::cce_integration_independent_tags,
                     typename Metavariables::cce_temporary_equations_tags>>;
    using pre_swsh_derivatives_variables_tag = ::Tags::Variables<
        typename Metavariables::cce_pre_swsh_derivatives_tags>;
    using transform_buffer_variables_tag =
        ::Tags::Variables<typename Metavariables::cce_transform_buffer_tags>;
    using swsh_derivative_variables_tag =
        ::Tags::Variables<typename Metavariables::cce_swsh_derivative_tags>;
    using angular_coordinates_variables_tag =
        ::Tags::Variables<typename Metavariables::cce_angular_coordinate_tags>;
    using coordinate_variables_tag =
        typename Metavariables::evolved_coordinates_variables_tag;
    using dt_coordinate_variables_tag =
        db::add_tag_prefix<::Tags::dt, coordinate_variables_tag>;
    using evolved_swsh_variables_tag =
        ::Tags::Variables<tmpl::list<typename Metavariables::evolved_swsh_tag>>;
    using evolved_swsh_dt_variables_tag = ::Tags::Variables<
        tmpl::list<typename Metavariables::evolved_swsh_dt_tag>>;
    template <typename TagList>
    static auto initialize(
        db::DataBox<TagList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
      const size_t l_max = Parallel::get<Spectral::Swsh::Tags::LMax>(cache);
      const size_t number_of_radial_points =
          Parallel::get<Spectral::Swsh::Tags::NumberOfRadialPoints>(cache);
      const size_t boundary_size =
          Spectral::Swsh::number_of_swsh_collocation_points(l_max);
      const size_t volume_size = boundary_size * number_of_radial_points;
      const size_t transform_buffer_size =
          number_of_radial_points *
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
      return Initialization::merge_into_databox<
          InitializeCharacteristicEvolution,
          db::AddSimpleTags<
              boundary_value_variables_tag, coordinate_variables_tag,
              dt_coordinate_variables_tag, evolved_swsh_variables_tag,
              evolved_swsh_dt_variables_tag, angular_coordinates_variables_tag,
              scri_variables_tag, volume_variables_tag,
              pre_swsh_derivatives_variables_tag,
              transform_buffer_variables_tag, swsh_derivative_variables_tag,
              Spectral::Swsh::Tags::SwshInterpolator<
                  Tags::CauchyAngularCoords>>,
          db::AddComputeTags<>, Initialization::MergePolicy::Overwrite>(
          std::move(box),
          db::item_type<boundary_value_variables_tag>{boundary_size},
          db::item_type<coordinate_variables_tag>{boundary_size},
          db::item_type<dt_coordinate_variables_tag>{boundary_size},
          db::item_type<evolved_swsh_variables_tag>{volume_size},
          db::item_type<evolved_swsh_dt_variables_tag>{volume_size},
          db::item_type<angular_coordinates_variables_tag>{boundary_size},
          db::item_type<scri_variables_tag>{boundary_size},
          db::item_type<volume_variables_tag>{volume_size},
          db::item_type<pre_swsh_derivatives_variables_tag>{volume_size, 0.0},
          db::item_type<transform_buffer_variables_tag>{transform_buffer_size,
                                                        0.0},
          db::item_type<swsh_derivative_variables_tag>{volume_size, 0.0},
          Spectral::Swsh::SwshInterpolator{});
    }
  };

  template <class Metavariables>
  using return_tag_list = tmpl::append<
      typename EvolutionTags<Metavariables>::evolution_simple_tags,
      typename EvolutionTags<Metavariables>::evolution_compute_tags>;

  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTags, InitializationTags::StartTime>> =
          nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto evolution_box =
        EvolutionTags<Metavariables>::initialize(std::move(box), cache);
    auto characteristic_evolution_box =
        CharacteristicTags<Metavariables>::initialize(std::move(evolution_box),
                                                      cache);
    auto initialization_moved_box =
        Initialization::merge_into_databox<InitializeCharacteristicEvolution,
                                           db::AddSimpleTags<Tags::EndTime>,
                                           db::AddComputeTags<>>(
            std::move(characteristic_evolution_box),
            db::get<InitializationTags::EndTime>(characteristic_evolution_box));
    return std::make_tuple(std::move(initialization_moved_box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                DbTags, InitializationTags::StartTime>> = nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "The DataBox is missing required dependency "
        "`Cce::InitializationTags::LMax.`");
    // return required for type inference, this code should be unreachable due
    // to the `ERROR` on the previous line
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Cce
