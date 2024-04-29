// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"

namespace Cce::TestHelpers {

template <typename EvolutionComponent, typename Metavariables>
void check_characteristic_initialization(
    ActionTesting::MockRuntimeSystem<Metavariables>& runner,
    const double start_time, const double target_step_size, const size_t l_max,
    const size_t number_of_radial_points) {
  // the tags inserted in the `EvolutionTags` step
  const auto& time_step_id =
      ActionTesting::get_databox_tag<EvolutionComponent, ::Tags::TimeStepId>(
          runner, 0);
  CHECK(time_step_id.substep_time() == start_time);
  const auto& next_time_step_id =
      ActionTesting::get_databox_tag<EvolutionComponent,
                                     ::Tags::Next<::Tags::TimeStepId>>(runner,
                                                                       0);
  CHECK(next_time_step_id.substep_time() ==
        approx(start_time + target_step_size * 0.75));
  const auto& time_step =
      ActionTesting::get_databox_tag<EvolutionComponent, ::Tags::TimeStep>(
          runner, 0);
  CHECK(time_step.value() == approx(target_step_size * 0.75));
  const auto& coordinates_history = ActionTesting::get_databox_tag<
      EvolutionComponent,
      ::Tags::HistoryEvolvedVariables<
          typename Metavariables::evolved_coordinates_variables_tag>>(runner,
                                                                      0);
  CHECK(coordinates_history.size() == 0);
  const auto& evolved_swsh_history = ActionTesting::get_databox_tag<
      EvolutionComponent, ::Tags::HistoryEvolvedVariables<::Tags::Variables<
                              typename Metavariables::evolved_swsh_tags>>>(
      runner, 0);
  CHECK(evolved_swsh_history.size() == 0);

  // the tensor storage variables inserted during the `CharacteristicTags` step
  const auto& boundary_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      ::Tags::Variables<
          tmpl::append<typename Metavariables::cce_boundary_communication_tags,
                       typename Metavariables::cce_gauge_boundary_tags>>>(
      runner, 0);
  CHECK(boundary_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& coordinate_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      typename Metavariables::evolved_coordinates_variables_tag>(runner, 0);
  CHECK(coordinate_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& dt_coordinate_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      db::add_tag_prefix<::Tags::dt, typename Metavariables::
                                         evolved_coordinates_variables_tag>>(
      runner, 0);
  CHECK(dt_coordinate_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& angular_coordinates_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      ::Tags::Variables<typename Metavariables::cce_angular_coordinate_tags>>(
      runner, 0);
  CHECK(angular_coordinates_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  const auto& scri_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      ::Tags::Variables<typename Metavariables::cce_scri_tags>>(runner, 0);
  CHECK(scri_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& volume_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      ::Tags::Variables<
          tmpl::append<typename Metavariables::cce_integrand_tags,
                       typename Metavariables::cce_integration_independent_tags,
                       typename Metavariables::cce_temporary_equations_tags>>>(
      runner, 0);
  CHECK(volume_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
            number_of_radial_points);

  const auto& evolved_swsh_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      ::Tags::Variables<typename Metavariables::evolved_swsh_tags>>(runner, 0);
  CHECK(evolved_swsh_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
            number_of_radial_points);

  const auto& evolved_swsh_dt_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      ::Tags::Variables<typename Metavariables::evolved_swsh_dt_tags>>(runner,
                                                                       0);
  CHECK(evolved_swsh_dt_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
            number_of_radial_points);

  const auto& pre_swsh_derivatives_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      ::Tags::Variables<typename Metavariables::cce_pre_swsh_derivatives_tags>>(
      runner, 0);
  const Variables<typename Metavariables::cce_pre_swsh_derivatives_tags>
      expected_zeroed_pre_swsh_derivatives{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
              number_of_radial_points,
          0.0};
  CHECK(pre_swsh_derivatives_variables == expected_zeroed_pre_swsh_derivatives);

  const auto& transform_buffer_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      ::Tags::Variables<typename Metavariables::cce_transform_buffer_tags>>(
      runner, 0);
  const Variables<typename Metavariables::cce_transform_buffer_tags>
      expected_zeroed_transform_buffer{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max) *
              number_of_radial_points,
          0.0};
  CHECK(transform_buffer_variables == expected_zeroed_transform_buffer);

  const auto& swsh_derivatives_variables = ActionTesting::get_databox_tag<
      EvolutionComponent,
      ::Tags::Variables<typename Metavariables::cce_swsh_derivative_tags>>(
      runner, 0);
  const Variables<typename Metavariables::cce_swsh_derivative_tags>
      expected_zeroed_swsh_derivatives{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
              number_of_radial_points,
          0.0};
  CHECK(swsh_derivatives_variables == expected_zeroed_swsh_derivatives);
}
}  // namespace Cce::TestHelpers
