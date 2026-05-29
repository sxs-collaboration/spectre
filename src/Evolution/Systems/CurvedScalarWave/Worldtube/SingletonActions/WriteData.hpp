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
#include "Domain/FunctionsOfTime/QuaternionWorldtubeFunctionOfTime.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"

namespace CurvedScalarWave::Worldtube {
/*!
 * \brief Writes all components of a function of time to disk at a specific time
 * from a control system after it updates the functions of time.
 *
 * \details The columns of data written are:
 * - %Time
 * - FunctionOfTime
 * - dtFunctionOfTime
 *
 * Data will be stored in the reduction file. All subfiles for the control
 * system within the H5 file will be under the group "/FunctionsOfTime".
 * Within this group, there will be one group for each control system. An
 * example would look like
 *
 * - /FunctionsOfTime/Rotation
 *
 * Then, within each system group, there will be one subfile for each component
 * of the function of time that is being controlled. The name of this subfile is
 * the name of the component. For example, if "SystemA" has 4 components, then
 * the subfiles would look like
 *
 * - /FunctionsOfTime/SystemA/0.dat
 * - /FunctionsOfTime/SystemA/1.dat
 * - /FunctionsOfTime/SystemA/2.dat
 * - /FunctionsOfTime/SystemA/3.dat
 */
template <typename Metavariables>
void write_components_to_disk(
    const double time, Parallel::GlobalCache<Metavariables>& cache,
    const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>&
        function_of_time) {
  auto& observer_writer_proxy = Parallel::get_parallel_component<
      observers::ObserverWriter<Metavariables>>(cache);

  std::array<DataVector, 2> function_at_current_time{};
  const auto* const quat_func_of_time = dynamic_cast<
      const domain::FunctionsOfTime::QuaternionWorldtubeFunctionOfTime<1>*>(
      function_of_time.get());
  if (quat_func_of_time == nullptr) {
    // Just call the usual `func_and_deriv` member.
    function_at_current_time = function_of_time->func_and_deriv(time);
  } else {
    // If we are working with a QuaternionFunctionOfTime, we aren't actually
    // controlling a quaternion. We are controlling a small change in angle
    // associated with the angular velocity in each direction. Because of this,
    // we want to write the component data of the thing we are actually
    // controlling. This is accessed by the `angle_func_and_2_derivs` member of
    // a QuaternionFunctionOfTime. Since `angle_func_and_2_derivs` is not a
    // virtual function of the FunctionOfTime base class, we need to down cast
    // the original function to a QuaternionFunctionOfTime.

    // function_at_current_time = quat_func_of_time->angle_func_and_deriv(time);

    // We want to check the quaternion components themselves for now
    function_at_current_time = quat_func_of_time->quat_func_and_deriv(time);
  }

  // There is a different subfile for each component, so loop over them.
  const size_t num_components = function_at_current_time[0].size();
  for (size_t i = 0; i < num_components; ++i) {
    const std::string subfile_name{"/FunctionsOfTime/Rotation/" +
                                   std::to_string(i)};
    std::vector<std::string> legend{"Time", "FunctionOfTime",
                                    "dtFunctionOfTime"};

    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        // Node 0 is always the writer
        observer_writer_proxy[0], subfile_name, std::move(legend),
        std::make_tuple(
            // clang-format off
            time,
            function_at_current_time[0][i],
            function_at_current_time[1][i])
        // clang-format on
    );
  }
}
}  // namespace CurvedScalarWave::Worldtube
