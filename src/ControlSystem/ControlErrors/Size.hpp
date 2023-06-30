// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>

#include "ControlSystem/ControlErrors/Size/Error.hpp"
#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/Interpolation/ZeroCrossingPredictor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace Frame {
struct Grid;
struct Distorted;
}  // namespace Frame
/// \endcond

namespace control_system::ControlErrors {
/*!
 * \brief Control error in the for the \f$l=0\f$ component of the
 * `domain::CoordinateMaps::TimeDependent::Shape` map.
 *
 * \details The goal of this control error is
 *
 * 1. Keep the excision sphere inside the horizon
 * 2. Maintain a fixed distance between the excision surface and the horizon
 * surface.
 * 3. Prevent the characteristic field \f$ u^-_{ab} \f$ associated with the
 * characteristic speed \f$ v_- \f$ in `gh::characteristic_speeds` from coming
 * into the domain.
 *
 * For a more detailed account of how this is accomplished, see
 * `control_system::size::State` and `control_system::size::control_error` which
 * this class calls.
 *
 * This class holds a `control_system::size::Info` and three different
 * `intrp::ZeroCrossingPredictor`s internally which are needed to calculate the
 * `control_system::size::control_error`. It also conforms to the
 * `control_system::protocols::ControlError` protocol.
 *
 * In addition to calculating the control error, if the
 * `control_system::Tags::WriteDataToDisk` tag inside the
 * `Parallel::GlobalCache` is true, then a diagnostic file named
 * `Diagnostics.dat` is also written to the same group that
 * `control_system::write_components_to_disk` would write the standard control
 * system output (`/ControlSystems/Size/`). The columns of this diagnostic file
 * are as follows (with a small explanation if the name isn't clear):
 *
 * - %Time
 * - ControlError
 * - StateNumber: Result of `control_system::size::State::number()`
 * - DiscontinuousChangeHasOccurred: 1.0 for true, 0.0 for false.
 * - FunctionOfTime
 * - dtFunctionOfTime
 * - HorizonCoef00
 * - dtHorizonCoef00
 * - MinDeltaR: The minimum of the `StrahlkorperGr::radial_distance` between the
 *   horizon and the excision surfaces.
 * - MinRelativeDeltaR: MinDeltaR divided by the `Strahlkorper::average_radius`
 *   of the horizon
 * - ControlErrorDeltaR: \f$ \dot{S}_{00} (\lambda_{00} -
 *   r_{\mathrm{excision}}^{\mathrm{grid}} / Y_{00}) / S_{00} -
 *   \dot{\lambda}_{00} \f$
 * - TargetCharSpeed
 * - MinCharSpeed
 * - MinComovingCharSpeed: Eq. 98 in \cite Hemberger2012jz
 * - CharSpeedCrossingTime: %Time at which the min char speed is predicted to
 *   cross zero and become negative (or 0.0 if that time is in the past).
 * - ComovingCharSpeedCrossingTime: %Time at which the min comoving char speed
 *   is predicted to cross zero and become negative (or 0.0 if that time is in
 *   the past).
 * - DeltaRCrossingTime: %Time at which the distance between the excision and
 *   horizon surfaces is predicted to be zero (or 0.0 if that time is in the
 *   past).
 * - SuggestedTimescale: A timescale for the `TimescaleTuner` suggested by one
 *   of the State%s (or 0.0 if no timescale was suggested)
 * - DampingTime
 */
template <::domain::ObjectLabel Horizon>
struct Size : tt::ConformsTo<protocols::ControlError> {
  static constexpr size_t expected_number_of_excisions = 1;

  using object_centers = domain::object_list<Horizon>;

  struct MaxNumTimesForZeroCrossingPredictor {
    // Int so we get proper bounds checking
    using type = int;
    static constexpr Options::String help{
        "The maximum number of times used to calculate the zero crossing of "
        "the char speeds."};
    static int lower_bound() { return 3; }
  };

  using options = tmpl::list<MaxNumTimesForZeroCrossingPredictor>;
  static constexpr Options::String help{
      "Computes the control error for size control. Will also write a "
      "diagnostics file if the control systems are allowed to write data to "
      "disk."};

  /*!
   * \brief Initializes the `intrp::ZeroCrossingPredictor`s with the number of
   * times passed in (default 3).
   *
   * \details The internal `control_system::size::Info::state` is initialized to
   * `control_system::size::States::Initial`. All zero crossing predictors are
   * initialized with a minimum number of times 3. The input argument
   * `max_times` represents the maximum number of times.
   */
  Size(const int max_times = 3);

  /// Returns the internal `control_system::size::Info::suggested_time_scale`. A
  /// std::nullopt means that no timescale is suggested.
  const std::optional<double>& get_suggested_timescale() const;

  /*!
   * \brief Check if the `control_system::size::control_error` has decided to
   * switch states. Returns the internal
   * `control_system::size::Info::discontinuous_change_has_occurred`.
   */
  bool discontinuous_change_has_occurred() const;

  /*!
   * \brief Reset the internal `control_system::size::Info` using
   * `control_system::size::Info::reset`.
   */
  void reset();

  void pup(PUP::er& p);

  /*!
   * \brief Actually computes the control error.
   *
   * \details The internal `control_system::size::Info::damping_time` is updated
   * to the minimum of the `TimescaleTuner::current_timescale()` that is passed
   * in. Also expects these queue tags to be in the `measurements` argument:
   *
   * - `StrahlkorperTags::Strahlkorper<Frame::Distorted>`
   * - `QueueTags::ExcisionSurface<Frame::Distorted>`
   * - `::Tags::dt<StrahlkorperTags::Strahlkorper<Frame::Distorted>>`
   * - `QueueTags::LapseOnExcisionSurface`
   * - `QueueTags::ShiftyQuantity<Frame::Distorted>`
   * - `QueueTags::SpatialMetricOnExcisionSurface<Frame::Distorted>`
   * - `QueueTags::InverseSpatialMetricOnExcisionSurface<Frame::Distorted>`
   *
   * \return DataVector should be of size 1
   */
  template <typename Metavariables, typename... TupleTags>
  DataVector operator()(const ::TimescaleTuner& tuner,
                        const Parallel::GlobalCache<Metavariables>& cache,
                        const double time,
                        const std::string& function_of_time_name,
                        const tuples::TaggedTuple<TupleTags...>& measurements) {
    const Domain<3>& domain = get<domain::Tags::Domain<3>>(cache);
    const auto& excision_spheres = domain.excision_spheres();
    const auto& excision_sphere =
        excision_spheres.at("ExcisionSphere" + get_output(Horizon));
    const auto& functions_of_time = get<domain::Tags::FunctionsOfTime>(cache);

    const auto& excision_quantities =
        tuples::get<QueueTags::SizeExcisionQuantities<Frame::Distorted>>(
            measurements);
    const auto& horizon_quantities =
        tuples::get<QueueTags::SizeHorizonQuantities<Frame::Distorted>>(
            measurements);

    const double grid_frame_excision_sphere_radius = excision_sphere.radius();
    const Strahlkorper<Frame::Distorted>& apparent_horizon =
        tuples::get<StrahlkorperTags::Strahlkorper<Frame::Distorted>>(
            horizon_quantities);
    const Strahlkorper<Frame::Distorted>& excision_surface =
        tuples::get<QueueTags::ExcisionSurface<Frame::Distorted>>(
            excision_quantities);
    const Strahlkorper<Frame::Distorted>& time_deriv_apparent_horizon =
        tuples::get<
            ::Tags::dt<StrahlkorperTags::Strahlkorper<Frame::Distorted>>>(
            horizon_quantities);
    const Scalar<DataVector>& lapse =
        tuples::get<QueueTags::LapseOnExcisionSurface>(excision_quantities);
    const tnsr::I<DataVector, 3, Frame::Distorted>& shifty_quantity =
        tuples::get<QueueTags::ShiftyQuantity<Frame::Distorted>>(
            excision_quantities);
    const tnsr::ii<DataVector, 3, Frame::Distorted>&
        spatial_metric_on_excision = tuples::get<
            QueueTags::SpatialMetricOnExcisionSurface<Frame::Distorted>>(
            excision_quantities);
    const tnsr::II<DataVector, 3, Frame::Distorted>&
        inverse_spatial_metric_on_excision = tuples::get<
            QueueTags::InverseSpatialMetricOnExcisionSurface<Frame::Distorted>>(
            excision_quantities);

    info_.damping_time = min(tuner.current_timescale());

    const size::ErrorDiagnostics error_diagnostics = size::control_error(
        make_not_null(&info_), make_not_null(&char_speed_predictor_),
        make_not_null(&comoving_char_speed_predictor_),
        make_not_null(&delta_radius_predictor_), time, apparent_horizon,
        excision_surface, grid_frame_excision_sphere_radius,
        time_deriv_apparent_horizon, lapse, shifty_quantity,
        spatial_metric_on_excision, inverse_spatial_metric_on_excision,
        functions_of_time.at(function_of_time_name));

    if (Parallel::get<control_system::Tags::WriteDataToDisk>(cache)) {
      auto& observer_writer_proxy = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache);

      Parallel::threaded_action<
          observers::ThreadedActions::WriteReductionDataRow>(
          observer_writer_proxy[0], subfile_name_, legend_,
          std::make_tuple(
              time, error_diagnostics.control_error,
              static_cast<double>(error_diagnostics.state_number),
              error_diagnostics.discontinuous_change_has_occurred ? 1.0 : 0.0,
              error_diagnostics.lambda_00, error_diagnostics.dt_lambda_00,
              error_diagnostics.horizon_00, error_diagnostics.dt_horizon_00,
              error_diagnostics.min_delta_r,
              error_diagnostics.min_relative_delta_r,
              error_diagnostics.control_error_delta_r,
              error_diagnostics.target_char_speed,
              error_diagnostics.min_char_speed,
              error_diagnostics.min_comoving_char_speed,
              error_diagnostics.char_speed_crossing_time,
              error_diagnostics.comoving_char_speed_crossing_time,
              error_diagnostics.delta_r_crossing_time,
              error_diagnostics.suggested_timescale,
              error_diagnostics.damping_timescale));
    }

    return DataVector{1, error_diagnostics.control_error};
  }

 private:
  size::Info info_{};
  intrp::ZeroCrossingPredictor char_speed_predictor_{};
  intrp::ZeroCrossingPredictor comoving_char_speed_predictor_{};
  intrp::ZeroCrossingPredictor delta_radius_predictor_{};
  std::vector<std::string> legend_{};
  std::string subfile_name_{};
};
}  // namespace control_system::ControlErrors
