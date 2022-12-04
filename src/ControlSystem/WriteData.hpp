// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "ControlSystem/Component.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"

namespace control_system {
/*!
 * \ingroup ControlSystemGroup
 * \brief Writes all components of a function of time to disk at a specific time
 * from a control system after it updates the functions of time.
 *
 * \details The columns of data written are:
 * - %Time
 * - Lambda
 * - dtLambda
 * - d2tLambda
 * - ControlError
 * - dtControlError
 *
 * where "Lambda" is the function of time value.
 *
 * Data will be stored in the reduction file. All subfiles for the control
 * system within the H5 file will be under the group "/ControlSystems".
 * Within this group, there will be one group for each control system. The name
 * of each group will be the result of the `name()` function from each control
 * system. An example would look like
 *
 * - /ControlSystems/SystemA
 * - /ControlSystems/SystemB
 * - /ControlSystems/SystemC
 *
 * Then, within each system group, there will be one subfile for each component
 * of the function of time that is being controlled. The name of this subfile is
 * the name of the component. The name of each component will be the result of
 * the `component_name(i)` function from the control system, where `i` is the
 * index of the component. For example, if "SystemA" has 3 components with names
 * "X", "Y", and "Z", then the subfiles would look like
 *
 * - /ControlSystems/SystemA/X.dat
 * - /ControlSystems/SystemA/Y.dat
 * - /ControlSystems/SystemA/Z.dat
 */
template <typename ControlSystem, typename Metavariables>
void write_components_to_disk(
    const double time, Parallel::GlobalCache<Metavariables>& cache,
    const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>&
        function_of_time,
    const std::array<DataVector, 2>& q_and_dtq) {
  auto& observer_writer_proxy = Parallel::get_parallel_component<
      observers::ObserverWriter<Metavariables>>(cache);

  constexpr size_t deriv_order = ControlSystem::deriv_order;
  std::array<DataVector, 3> function_at_current_time{};
  const auto* const quat_func_of_time = dynamic_cast<
      const domain::FunctionsOfTime::QuaternionFunctionOfTime<deriv_order>*>(
      function_of_time.get());
  if (quat_func_of_time == nullptr) {
    // Just call the usual `func_and_2_derivs` member.
    function_at_current_time = function_of_time->func_and_2_derivs(time);
  } else {
    // If we are working with a QuaternionFunctionOfTime, we aren't actually
    // controlling a quaternion. We are controlling a small change in angle
    // associated with the angular velocity in each direction. Because of this,
    // we want to write the component data of the thing we are actually
    // controlling. This is accessed by the `angle_func_and_2_derivs` member of
    // a QuaternionFunctionOfTime. Since `angle_func_and_2_derivs` is not a
    // virtual function of the FunctionOfTime base class, we need to down cast
    // the original function to a QuaternionFunctionOfTime.
    function_at_current_time = quat_func_of_time->angle_func_and_2_derivs(time);
  }

  // Each control system is its own group under the overarching `ControlSystems`
  // group. There is a different subfile for each component, so loop over them.
  // Eg. translation files will look like
  //
  // ControlSystems/Translation/X.dat
  // ControlSystems/Translation/Y.dat
  // ControlSystems/Translation/Z.dat
  const size_t num_components = function_at_current_time[0].size();
  for (size_t i = 0; i < num_components; ++i) {
    const std::optional<std::string> component_name_opt =
        ControlSystem::component_name(i, num_components);
    if (not component_name_opt) {
      continue;
    }
    // Currently all reduction data is written to the reduction file so preface
    // everything with ControlSystems/
    const std::string subfile_name{"/ControlSystems/" + ControlSystem::name() +
                                   "/" + *component_name_opt};
    std::vector<std::string> legend{"Time",         "Lambda",
                                    "dtLambda",     "d2tLambda",
                                    "ControlError", "dtControlError"};

    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        // Node 0 is always the writer
        observer_writer_proxy[0], subfile_name, std::move(legend),
        std::make_tuple(
            // clang-format off
            time,
            function_at_current_time[0][i],
            function_at_current_time[1][i],
            function_at_current_time[2][i],
            q_and_dtq[0][i],
            q_and_dtq[1][i])
        // clang-format on
    );
  }
}
}  // namespace control_system
