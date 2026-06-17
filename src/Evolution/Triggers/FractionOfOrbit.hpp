// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iostream>
#include <memory>
#include <pup.h>
#include <string>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Callback.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::Tags {
template <size_t VolumeDim>
struct Domain;
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace Tags {
struct Time;
}  // namespace Tags

namespace DenseTriggers {
class FractionOfOrbit : public DenseTrigger {
 public:
  /// \cond
  FractionOfOrbit() = default;
  explicit FractionOfOrbit(CkMigrateMessage* const msg) : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FractionOfOrbit);  // NOLINT
  /// \endcond

  struct Value {
    using type = double;
    static constexpr Options::String help = {
        "Fraction of an orbit completed between triggers."};
  };

  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
  };

  using options = tmpl::list<Value, InitialTime>;
  static constexpr Options::String help{
      "Trigger at a fraction of an orbit since last trigger."};

  explicit FractionOfOrbit(double fraction_of_orbit, double initial_time);

  using is_triggered_return_tags = tmpl::list<>;
  using is_triggered_argument_tags =
      tmpl::list<Tags::Time, domain::Tags::FunctionsOfTime>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<bool> is_triggered(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const Component* /*component*/,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) {
    return is_triggered_impl(time, functions_of_time);
  }

  using next_check_time_return_tags = tmpl::list<>;
  using next_check_time_argument_tags =
      tmpl::list<Tags::Time, domain::Tags::FunctionsOfTime>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ParallelComponent* const /*meta*/,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) {
    for (auto i = functions_of_time.begin(); i != functions_of_time.end();
         i++) {
      const auto* const rot_f_of_t = dynamic_cast<
          const domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
          (i->second.get()));
      if (rot_f_of_t != nullptr) {
        const auto& this_proxy =
            ::Parallel::get_parallel_component<ParallelComponent>(
                cache)[array_index];
        bool ready = rot_f_of_t->time_bounds()[1] > time;
        // Calculate group_expiration
        // Parallel::mutable_cache_item_is_ready<domain::Tags::FunctionsOfTime>(
        //     cache,
        //     Parallel::make_array_component_id<ParallelComponent>
        // (array_index),
        //     [&](const auto& functions_of_time) {
        //       return ready ? std::unique_ptr<Parallel::Callback>{}
        //                    : std::unique_ptr<Parallel::Callback>(
        //                          new Parallel::PerformAlgorithmCallback(
        //                              this_proxy));
        //     });
        if (not ready) {
          return std::nullopt;
        }
        if (last_trigger_time_ > time) {
          return last_trigger_time_;
        }
        const double expiration_time = rot_f_of_t->expiration_after(time);
        const double sin_func_in_quat =
            sin(acos(rot_f_of_t->quat_func_and_deriv(time)[0][0]));
        if (sin_func_in_quat == 0) {
          return expiration_time;
        }
        const double omega =
            abs(2.0 * rot_f_of_t->quat_func_and_deriv(time)[1][0] /
                sin_func_in_quat);
        const double angle_left_to_orbit =
            2 * M_PI * fraction_of_orbit_ -
            abs(rot_f_of_t->full_angle(time) -
                rot_f_of_t->full_angle(last_trigger_time_));
        const double next_time = time + (angle_left_to_orbit / omega);
        if (next_time < time) {
          std::cout << "Is this the problem?";
          return expiration_time;
        }
        std::cout << "expiration time: " << expiration_time;
        std::cout << "next time: " << next_time;
        if (expiration_time < next_time) {
          return expiration_time;
        } else {
          return next_time;
        }
      }
    }
    ERROR(
        "FractionOfOrbit trigger can only be used when the rotation map is "
        "active");
  }

  // NOLINENEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::optional<bool> is_triggered_impl(
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time);

  double fraction_of_orbit_{};
  double last_trigger_time_{};
};
}  // namespace DenseTriggers
