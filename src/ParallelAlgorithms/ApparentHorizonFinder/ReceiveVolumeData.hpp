// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/InvokeCallbacks.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/CleanUp.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputePoints.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/CurrentTime.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolateVolumeVars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
template <typename HorizonMetavars>
struct ReceiveVolumeData {
  using frame = typename HorizonMetavars::frame;

  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const LinkedMessageId<double>& incoming_time,
                    const ElementId<3>& incoming_element_id,
                    const ::Mesh<3>& incoming_mesh,
                    Variables<ah::source_vars<3>>&& incoming_source_vars,
                    const std::optional<std::string>& dependency,
                    const bool source_vars_have_already_been_received = false) {
    const auto& verbosity = db::get<ah::Tags::Verbosity>(box);
    const bool quiet_print = verbosity >= ::Verbosity::Quiet;
    const bool verbose_print = verbosity >= ::Verbosity::Verbose;
    const bool debug_print = verbosity >= ::Verbosity::Debug;
    const std::string& name = pretty_type::name<HorizonMetavars>();

    // If we've already completed this horizon find, ignore the volume data and
    // just return
    auto& completed_times = db::get_mutable_reference<ah::Tags::CompletedTimes>(
        make_not_null(&box));
    if (completed_times.contains(incoming_time)) {
      if (debug_print) {
        Parallel::printf(
            "%s: Received volume data at finished time %s from Element %s. "
            "Ignoring.\n",
            name, incoming_time, incoming_element_id);
      }
      return;
    }

    auto& current_time_optional =
        db::get_mutable_reference<ah::Tags::CurrentTime>(make_not_null(&box));
    auto& pending_times =
        db::get_mutable_reference<ah::Tags::PendingTimes>(make_not_null(&box));

    // Only add to pending if it isn't the current time.
    if (not(current_time_optional.has_value() and
            current_time_optional.value() == incoming_time)) {
      pending_times.insert(incoming_time);
    }

    auto& all_storage = db::get_mutable_reference<ah::Tags::Storage<frame>>(
        make_not_null(&box));

    // Add volume variables and destination to the box if they haven't already
    // been received
    if (not source_vars_have_already_been_received) {
      all_storage[incoming_time].all_volume_variables.emplace(
          incoming_element_id,
          ah::Storage::VolumeVariables<frame>{incoming_mesh,
                                              std::move(incoming_source_vars)});
      all_storage.at(incoming_time).destination = HorizonMetavars::destination;
    }

    // ========================================================================
    // The incoming quantities are only used in the above code. Below, we only
    // use quantities at the current time
    // ========================================================================

    // Keep trying to find horizons for as long as we can
    while (true) {
      // Set current time using the pending times if it isn't already set
      set_current_time(make_not_null(&current_time_optional),
                       make_not_null(&pending_times), completed_times,
                       all_storage, verbosity, name);

      // At this point, if we don't have a current id we can't do anything so
      // just exit.
      if (not current_time_optional.has_value()) {
        if (verbose_print) {
          Parallel::printf(
              "%s: No current time set. Stopping here. Incoming time: %s\n",
              name, incoming_time);
        }
        return;
      }

      const LinkedMessageId<double>& current_time =
          current_time_optional.value();
      ASSERT(all_storage.contains(current_time),
             "Current time doesn't have any data.");

      auto& current_time_storage = all_storage.at(current_time);

      // Only check if the current time is ready once
      if (UNLIKELY(not current_time_storage.time_is_ready)) {
        if (check_if_current_time_is_ready<HorizonMetavars>(
                current_time, cache, incoming_time, incoming_element_id,
                incoming_mesh, dependency, verbosity)) {
          // Set this so we don't check it again
          current_time_storage.time_is_ready = true;

          if (verbose_print) {
            Parallel::printf("%s: Current time %s is ready.\n", name,
                             current_time);
          }
        } else {
          if (verbose_print) {
            Parallel::printf("%s: Current time %s is NOT ready.\n", name,
                             current_time);
          }
          // We need to wait for the functions of time to be updated so we can't
          // do anything yet.
          return;
        }
      }

      const auto& domain = Parallel::get<domain::Tags::Domain<3>>(cache);
      const auto& functions_of_time =
          Parallel::get<domain::Tags::FunctionsOfTime>(cache);
      const auto& options =
          Parallel::get<ah::Tags::ApparentHorizonOptions<HorizonMetavars>>(
              cache);
      auto& fast_flow =
          db::get_mutable_reference<ah::Tags::FastFlow>(make_not_null(&box));

      auto& all_volume_variables = current_time_storage.all_volume_variables;
      auto& previous_surfaces =
          db::get_mutable_reference<ah::Tags::PreviousSurfaces<frame>>(
              make_not_null(&box));

      // Keep doing fast flow iterations for as long as we can until we find a
      // horizon
      while (true) {
        auto& current_iteration_storage =
            current_time_storage.current_iteration;
        auto& previous_iteration_surface =
            current_time_storage.previous_iteration_surface;

        // Points haven't been set for this iteration so need to do so now
        if (not current_iteration_storage.block_coord_holders.has_value()) {
          const bool coords_set_successfully = set_current_iteration_coords(
              make_not_null(&current_iteration_storage), current_time,
              fast_flow, options.initial_guess, previous_iteration_surface,
              previous_surfaces, options.max_interpolation_retries, domain,
              functions_of_time);

          // Some points are outside the domain and we cannot recover
          if (not coords_set_successfully) {
            // This will possibly throw an error
            tmpl::for_each<
                typename HorizonMetavars::horizon_find_failure_callbacks>(
                [&](auto callback_v) {
                  using callback = tmpl::type_from<decltype(callback_v)>;
                  callback::apply(box, cache,
                                  FastFlow::Status::InterpolationFailure);
                });

            // Still clean up in case we didn't error so we don't keep accepting
            // volume data for this time
            clean_up_horizon_finder(make_not_null(&current_time_optional),
                                    make_not_null(&all_storage),
                                    make_not_null(&completed_times),
                                    make_not_null(&fast_flow));

            // If we didn't error, move on to the next horizon find
            break;
          }

          if (debug_print) {
            Parallel::printf(
                "%s: For current time %s, at iteration %u, coords have been "
                "set.\n",
                name, current_time, fast_flow.current_iteration());
          }
        }

        // Interpolate new elements to points
        interpolate_volume_data(make_not_null(&current_iteration_storage),
                                make_not_null(&all_volume_variables),
                                current_time, domain, functions_of_time);

        // Sanity check
        ASSERT(
            current_iteration_storage.indicies_interpolated_to_thus_far
                    .size() <=
                current_iteration_storage.block_coord_holders.value().size(),
            "The number indices interpolated to is larger than the number of "
            "coordinates. This is an internal error. Please file an issue.");

        if (current_iteration_storage.indicies_interpolated_to_thus_far
                .size() !=
            current_iteration_storage.block_coord_holders.value().size()) {
          if (debug_print) {
            Parallel::printf(
                "%s: For current time %s, at iteration %u, need more volume "
                "data. Missing %u points.\n",
                name, current_time, fast_flow.current_iteration(),
                current_iteration_storage.block_coord_holders.value().size() -
                    current_iteration_storage.indicies_interpolated_to_thus_far
                        .size());
          }
          return;
        }

        // Set this before we iterate the fast flow. Note that this will have to
        // be updated once we have adaptive horizon finding
        previous_iteration_surface = current_iteration_storage.strahlkorper;

        // Do a single fast flow iteration.
        std::pair<FastFlow::Status, FastFlow::IterInfo> status_and_info;
        {
          using InverseSpatialMetric =
              gr::Tags::InverseSpatialMetric<DataVector, 3, frame>;
          using ExtrinsicCurvature =
              gr::Tags::ExtrinsicCurvature<DataVector, 3, frame>;
          using SpatialChristoffel =
              gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, frame>;

          auto& interpolated_vars = current_iteration_storage.interpolated_vars;
          status_and_info = fast_flow.iterate_horizon_finder(
              make_not_null(&current_iteration_storage.strahlkorper),
              get<InverseSpatialMetric>(interpolated_vars),
              get<ExtrinsicCurvature>(interpolated_vars),
              get<SpatialChristoffel>(interpolated_vars));
        }

        const auto& status = status_and_info.first;
        const auto& info = status_and_info.second;
        const auto has_converged = converged(status);

        if (verbose_print or (quiet_print and has_converged)) {
          Parallel::printf(
              "%s: t=%.6g: its=%d: %.1e<R<%.0e, |R|=%.1g, "
              "|R_grid|=%.1g, %.4g<r<%.4g\n",
              pretty_type::name<HorizonMetavars>(), current_time.id,
              info.iteration, info.min_residual, info.max_residual,
              info.residual_ylm, info.residual_mesh, info.r_min, info.r_max);
        }

        if (status == FastFlow::Status::SuccessfulIteration) {
          // Reset so we compute and interpolate to new points
          current_iteration_storage.reset_for_next_iteration();

          // Continue in the iteration while loop
          continue;
        } else if (not has_converged) {
          // This will possibly throw an error
          tmpl::for_each<
              typename HorizonMetavars::horizon_find_failure_callbacks>(
              [&](auto callback_v) {
                using callback = tmpl::type_from<decltype(callback_v)>;
                callback::apply(box, cache, status);
              });

          // Still clean up in case we didn't error so we don't keep accepting
          // volume data for this time
          clean_up_horizon_finder(make_not_null(&current_time_optional),
                                  make_not_null(&all_storage),
                                  make_not_null(&completed_times),
                                  make_not_null(&fast_flow));

          // If we didn't error, move on to the next horizon find
          break;
        }

        // We have converged to the apparent horizon. Invoke the callbacks and
        // clean up for the next horizon find
        invoke_callbacks<HorizonMetavars>(make_not_null(&box), cache,
                                          dependency, status);

        if (debug_print) {
          Parallel::printf(
              "%s: Horizon find finished at current time %s. Cleaning up and "
              "moving to next time\n",
              name, current_time);
        }

        clean_up_horizon_finder(
            make_not_null(&current_time_optional), make_not_null(&all_storage),
            make_not_null(&completed_times), make_not_null(&fast_flow));

        // Break out of the iteration while loop. Go back into the overall while
        // loop for time
        break;
      }  // while (true)
    }    // while (true)
  }
};
}  // namespace ah
