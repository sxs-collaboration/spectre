// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace control_system::Tags {
struct UpdateAggregators;
struct SystemToCombinedNames;
struct MeasurementTimescales;
}  // namespace control_system::Tags
/// \endcond

namespace control_system {
/// \ingroup ControlSystemGroup
/// Updates a FunctionOfTime in the global cache. Intended to be used in
/// Parallel::mutate.
struct UpdateSingleFunctionOfTime {
  static void apply(
      gsl::not_null<std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
          f_of_t_list,
      const std::string& f_of_t_name, double update_time,
      DataVector update_deriv, double new_expiration_time);
};

/*!
 * \ingroup ControlSystemGroup
 * \brief Updates several FunctionOfTimes in the global cache at once. Intended
 * to be used in Parallel::mutate.
 *
 * \details All functions of time are updated at the same `update_time`. For the
 * `update_args`, the keys of the map are the names of the functions of time.
 * The value `std::pair<DataVector, double>` for each key is the updated
 * derivative for the function of time and the new expiration time,
 * respectively.
 */
struct UpdateMultipleFunctionsOfTime {
  static void apply(
      gsl::not_null<std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
          f_of_t_list,
      double update_time,
      const std::unordered_map<std::string, std::pair<DataVector, double>>&
          update_args);
};

/// \ingroup ControlSystemGroup
/// Resets the expiration time of a FunctionOfTime in the global cache. Intended
/// to be used in Parallel::mutate.
struct ResetFunctionOfTimeExpirationTime {
  static void apply(
      gsl::not_null<std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
          f_of_t_list,
      const std::string& f_of_t_name, double new_expiration_time);
};

/*!
 * \brief A class for collecting and storing information related to updating
 * functions of time and measurement timescales.
 *
 * \details This class also determines if enough data has been received in order
 * for the functions of time and measurement timescales to be updated. There
 * should be one `UpdateAggregator` for every group of control systems that have
 * the same `control_system::protocols::Measurement`.
 */
struct UpdateAggregator {
  UpdateAggregator() = default;

  /*!
   * \brief Construct a new UpdateAggregator using a set of active control
   * system names and the combined name for the measurement.
   *
   * It is expected that all the control systems in this set use the same
   * `control_system::protocols::Measurement`.
   */
  UpdateAggregator(std::string combined_name,
                   std::unordered_set<std::string> active_control_system_names);

  /*!
   * \brief Inserts and stores information for one of the control systems that
   * this class was constructed with.
   *
   * \param control_system_name Name of control system to add information for
   * \param new_measurement_timescale DataVector of new measurement timescales
   * calculated from a control system update.
   * \param new_measurement_expiration_time New measurement expiration time
   * calculated during that update.
   * \param control_signal New highest derivative for the function of time
   * calculated during that update (will be `std::move`ed).
   * \param new_fot_expiration_time New function of time expiration time
   * calculated for during that update
   */
  void insert(const std::string& control_system_name,
              const DataVector& new_measurement_timescale,
              double new_measurement_expiration_time, DataVector control_signal,
              double new_fot_expiration_time);

  /*!
   * \brief Checks if `insert` has been called for all control systems that this
   * class was constructed with.
   */
  bool is_ready() const;

  /*!
   * \brief Returns a sorted concatenation of the control system names this
   * class was constructed with.
   */
  const std::string& combined_name() const;

  /*!
   * \brief Once `is_ready` is true, returns a map between the control system
   * name and a `std::pair` containing the `control_signal` that was passed to
   * `insert` and the minimum of all the `new_fot_expiration_time`s passed to
   * `insert`.
   *
   * \details This function is expected to only be called when `is_ready` is
   * true. It also must be called before `combined_measurement_expiration_time`.
   */
  std::unordered_map<std::string, std::pair<DataVector, double>>
  combined_fot_expiration_times() const;

  /*!
   * \brief Once `is_ready` is true, returns a `std::pair` containing the
   * minimum of all `new_measurement_timescale`s passed to `insert` and the
   * minimum of all `new_measurement_expiration_time`s passed to `insert`.
   *
   * \details This function is expected to be called only when `is_ready` is
   * true and only a single time once all control active control
   * systems for this measurement have computed their update values. It also
   * must be called after `combined_fot_expiration_times`. This function clears
   * all stored data when it is called.
   */
  std::pair<double, double> combined_measurement_expiration_time();

  /// \cond
  void pup(PUP::er& p);
  /// \endcond

 private:
  std::unordered_map<std::string, std::pair<std::pair<DataVector, double>,
                                            std::pair<double, double>>>
      expiration_times_{};
  std::unordered_set<std::string> active_names_{};
  std::string combined_name_{};
};

/*!
 * \brief Simple action that updates the appropriate `UpdateAggregator` for the
 * templated `ControlSystem`.
 *
 * \details The `new_measurement_timescale`, `new_measurement_expiration_time`,
 * `control_signal`, and `new_fot_expiration_time` are passed along to the
 * `UpdateAggregator::insert` function. The `old_measurement_expiration_time`
 * and `old_fot_expiration_time` are only used when the
 * `UpdateAggregator::is_ready` in order to update the functions of time and the
 * measurement timescale.
 *
 * When the `UpdateAggregator::is_ready`, the measurement timescale is mutated
 * with `UpdateSingleFunctionOfTime` and the functions of time are mutated with
 * `UpdateMultipleFunctionsOfTime`, both using `Parallel::mutate`.
 *
 * The "appropriate" `UpdateAggregator` is chosen from the
 * `control_system::Tags::SystemToCombinedNames` for the templated
 * `ControlSystem`.
 */
template <typename ControlSystem>
struct AggregateUpdate {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const DataVector& new_measurement_timescale,
                    const double old_measurement_expiration_time,
                    const double new_measurement_expiration_time,
                    DataVector control_signal,
                    const double old_fot_expiration_time,
                    const double new_fot_expiration_time) {
    auto& aggregators =
        db::get_mutable_reference<Tags::UpdateAggregators>(make_not_null(&box));
    const auto& system_to_combined_names =
        Parallel::get<Tags::SystemToCombinedNames>(cache);
    const std::string& control_system_name = ControlSystem::name();
    ASSERT(system_to_combined_names.count(control_system_name) == 1,
           "Expected name '" << control_system_name
                             << "' to be in map of system-to-combined names, "
                                "but it wasn't. Keys are: "
                             << keys_of(system_to_combined_names));
    const std::string& combined_name =
        system_to_combined_names.at(control_system_name);
    ASSERT(aggregators.count(combined_name) == 1,
           "Expected combined name '" << combined_name
                                      << "' to be in map of aggregators, "
                                         "but it wasn't. Keys are: "
                                      << keys_of(aggregators));

    UpdateAggregator& aggregator = aggregators.at(combined_name);

    aggregator.insert(control_system_name, new_measurement_timescale,
                      new_measurement_expiration_time,
                      std::move(control_signal), new_fot_expiration_time);

    if (aggregator.is_ready()) {
      std::unordered_map<std::string, std::pair<DataVector, double>>
          combined_fot_expiration_times =
              aggregator.combined_fot_expiration_times();
      const std::pair<double, double> combined_measurement_expiration_time =
          aggregator.combined_measurement_expiration_time();

      Parallel::mutate<Tags::MeasurementTimescales, UpdateSingleFunctionOfTime>(
          cache, combined_name, old_measurement_expiration_time,
          DataVector{1, combined_measurement_expiration_time.first},
          combined_measurement_expiration_time.second);

      Parallel::mutate<::domain::Tags::FunctionsOfTime,
                       UpdateMultipleFunctionsOfTime>(
          cache, old_fot_expiration_time,
          std::move(combined_fot_expiration_times));
    }
  }
};
}  // namespace control_system
