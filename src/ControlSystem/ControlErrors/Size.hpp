// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/ControlErrors/Size/Error.hpp"
#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "ControlSystem/ControlErrors/Size/StateHistory.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/Interpolation/ZeroCrossingPredictor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
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

namespace control_system {
namespace size {
/*!
 * \brief Function that computes the control error for
 * `control_system::size::States::DeltaR`.
 *
 * This is helpful to have calculated separately because other control errors
 * may make use of this quantity. The equation for the control error is given in
 * Eq. 96 in \cite Hemberger2012jz.
 *
 * \param horizon_00 The $l=0,m=0$ coefficient of the apparent horizon in the
 * distorted frame.
 * \param dt_horizon_00 The $l=0,m=0$ coefficient of the time derivative of the
 * apparent horizon in the distorted frame, where the derivative is taken in the
 * distorted frame as well.
 * \param lambda_00 The $l=0,m=0$ component of the function of time for the time
 * dependent map
 * \param dt_lambda_00 The $l=0,m=0$ component of the time derivative of the
 * function of time for the time dependent map
 * \param grid_frame_excision_sphere_radius Radius of the excision sphere in the
 * grid frame
 */
double control_error_delta_r(const double horizon_00,
                             const double dt_horizon_00, const double lambda_00,
                             const double dt_lambda_00,
                             const double grid_frame_excision_sphere_radius);
}  // namespace size

namespace ControlErrors {
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
 * `control_system::size::control_error`. Additionally, this class stores a
 * history of control errors for all `control_system::size::State`s using a
 * `control_system::size::StateHistory`. This is useful for when a discontinuous
 * change happens (switching `control_system::size::State`s) and we need to
 * repopulate the `Averager` with a history of the control error. It also
 * conforms to the `control_system::protocols::ControlError` protocol.
 *
 * In order to calculate the control error, we need the $\ell = 0, m = 0$
 * coefficient of the horizon and its time derivative. However, because we will
 * be finding the horizon fairly often, the value of the coefficient and its
 * derivative won't change smoothly because of horizon finder noise (different
 * number of iterations). But we expect these quantities to be smooth when
 * making state decisions. So to account for this, we use an `Averager` and a
 * `TimescaleTuner` to smooth out the horizon coefficient and get its
 * derivative. Every measurement, we update this smoothing averager with the
 * $\ell = 0, m = 0$ coefficient of the horizon and the current smoothing
 * timescale. Then, once we have enough measurements, we use the
 * `Averager::operator()` to get the averaged coefficient and its time
 * derivative. Since `Averager%s` calculate the average at an "averaged time",
 * we have to account for this small offset from the current time with a simple
 * Taylor expansion. Then we use this newly averaged and corrected coefficient
 * (and time derivative) in our calculation of the control error. The timescale
 * in the smoothing `TimescaleTuner` is then updated using the difference
 * between the averaged and un-averaged coefficient (and its time derivative).
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
 * - DtFunctionOfTime
 * - HorizonCoef00
 * - AveragedDtHorizonCoef00: The averaged 00 component of the horizon
 *   (averaging scheme detailed above.)
 * - RawDtHorizonCoef00: The raw 00 component of the horizon passed in to the
 *   control error.
 * - SmootherTimescale: Damping timescale for the averaging of DtHorizonCoef00.
 * - MinDeltaR: The minimum of the `gr::surfaces::radial_distance` between the
 *   horizon and the excision surfaces.
 * - MinRelativeDeltaR: MinDeltaR divided by the
 *   `ylm::Strahlkorper::average_radius` of the horizon
 * - AvgDeltaR: Same as MinDeltaR except it's the average radii.
 * - AvgRelativeDeltaR: AvgDeltaR divided by the average radius of the horizon
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
template <size_t DerivOrder, ::domain::ObjectLabel Horizon>
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

  struct SmoothAvgTimescaleFraction {
    using type = double;
    static constexpr Options::String help{
        "Average timescale fraction for smoothing horizon measurements."};
  };

  struct SmootherTuner {
    using type = TimescaleTuner<true>;
    static constexpr Options::String help{
        "TimescaleTuner for smoothing horizon measurements."};
  };

  struct InitialState {
    using type = std::unique_ptr<size::State>;
    static constexpr Options::String help{"Initial state to start in."};
  };

  struct DeltaRDriftOutwardOptions {
    using type =
        Options::Auto<DeltaRDriftOutwardOptions, Options::AutoLabel::None>;
    static constexpr Options::String help{
        "Options for State DeltaRDriftOutward. Specify 'None' to disable State "
        "DeltaRDriftOutward."};
    struct MaxAllowedRadialDistance {
      using type = double;
      static constexpr Options::String help{
          "Drift excision boundary outward if distance from horizon to "
          "excision exceeds this."};
    };
    struct OutwardDriftVelocity {
      using type = double;
      static constexpr Options::String help{
          "Constant drift velocity term, if triggered by "
          "MaxAllowedRadialDistance."};
    };
    struct OutwardDriftTimescale {
      using type = double;
      static constexpr Options::String help{
          "Denominator in non-constant drift velocity term, if triggered by "
          "MaxAllowedRadialDistance."};
    };
    using options = tmpl::list<MaxAllowedRadialDistance, OutwardDriftVelocity,
                               OutwardDriftTimescale>;
    void pup(PUP::er& p) {
      p | max_allowed_radial_distance;
      p | outward_drift_velocity;
      p | outward_drift_timescale;
    }

    double max_allowed_radial_distance{};
    double outward_drift_velocity{};
    double outward_drift_timescale{};
  };

  using options = tmpl::list<MaxNumTimesForZeroCrossingPredictor,
                             SmoothAvgTimescaleFraction, SmootherTuner,
                             InitialState, DeltaRDriftOutwardOptions>;
  static constexpr Options::String help{
      "Computes the control error for size control. Will also write a "
      "diagnostics file if the control systems are allowed to write data to "
      "disk."};

  Size() = default;

  /*!
   * \brief Initializes the `intrp::ZeroCrossingPredictor`s and the horizon
   * smoothing `Averager` and `TimescaleTuner`.
   *
   * \details All `intrp::ZeroCrossingPredictor`s are initialized with a minimum
   * number of times 3 and a maximum number of times `max_times`. The internal
   * `control_system::size::Info::state` is initialized to
   * `control_system::size::States::Initial`. The smoothing `Averager` uses the
   * input average timescale fraction and always smooths the "0th" deriv (aka
   * the horizon coefficients themselves). The input smoothing `TimescaleTuner`
   * is moved inside this class.
   */
  Size(const int max_times, const double smooth_avg_timescale_frac,
       TimescaleTuner<true> smoother_tuner,
       std::unique_ptr<size::State> initial_state,
       std::optional<DeltaRDriftOutwardOptions> delta_r_drift_outward_options);

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

  /*!
   * \brief Get a history of the control errors for the past few measurements.
   *
   * \return std::deque<std::pair<double, double>> This returns up to
   * `DerivOrder` entries, not including the most recent time. \see
   * `control_system::size::StateHistory::state_history`
   */
  std::deque<std::pair<double, double>> control_error_history() const;

  void pup(PUP::er& p);

  /*!
   * \brief Actually computes the control error.
   *
   * \details The internal `control_system::size::Info::damping_time` is updated
   * to the minimum of the `TimescaleTuner::current_timescale()` that is passed
   * in. Also expects these queue tags to be in the `measurements` argument:
   *
   * - `ylm::Tags::Strahlkorper<Frame::Distorted>`
   * - `QueueTags::ExcisionSurface<Frame::Distorted>`
   * - `::Tags::dt<ylm::Tags::Strahlkorper<Frame::Distorted>>`
   * - `QueueTags::LapseOnExcisionSurface`
   * - `QueueTags::ShiftyQuantity<Frame::Distorted>`
   * - `QueueTags::SpatialMetricOnExcisionSurface<Frame::Distorted>`
   * - `QueueTags::InverseSpatialMetricOnExcisionSurface<Frame::Distorted>`
   *
   * \return DataVector should be of size 1
   */
  template <typename Metavariables, typename... TupleTags>
  DataVector operator()(const ::TimescaleTuner<false>& tuner,
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
    const ylm::Strahlkorper<Frame::Distorted>& apparent_horizon =
        tuples::get<ylm::Tags::Strahlkorper<Frame::Distorted>>(
            horizon_quantities);
    const ylm::Strahlkorper<Frame::Distorted>& excision_surface =
        tuples::get<QueueTags::ExcisionSurface<Frame::Distorted>>(
            excision_quantities);
    const ylm::Strahlkorper<Frame::Distorted>& time_deriv_apparent_horizon =
        tuples::get<::Tags::dt<ylm::Tags::Strahlkorper<Frame::Distorted>>>(
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

    const double Y00 = 0.25 * M_2_SQRTPI;

    horizon_coef_averager_.update(time, {apparent_horizon.coefficients()[0]},
                                  smoother_tuner_.current_timescale());

    const std::optional<std::array<DataVector, DerivOrder + 1>>&
        averaged_horizon_coef_at_average_time = horizon_coef_averager_(time);

    // lambda_00 is the quantity of the same name in ArXiv:1211.6079,
    // and dt_lambda_00 is its time derivative.
    // This is the map parameter that maps the excision boundary in the grid
    // frame to the excision boundary in the distorted frame.
    const auto map_lambda_and_deriv =
        functions_of_time.at(function_of_time_name)->func_and_deriv(time);
    const double lambda_00 = map_lambda_and_deriv[0][0];
    const double dt_lambda_00 = map_lambda_and_deriv[1][0];

    // horizon_00 is \hat{S}_00 in ArXiv:1211.6079,
    // and dt_horizon_00 is its time derivative.
    // These are coefficients of the horizon in the distorted frame. However, we
    // want them averaged
    double horizon_00 = apparent_horizon.coefficients()[0];
    double dt_horizon_00 = time_deriv_apparent_horizon.coefficients()[0];

    // Only for the first few measurements will this not have a value
    if (LIKELY(averaged_horizon_coef_at_average_time.has_value())) {
      // We need to get the averaged time and evaluate the averaged coefs at
      // that time, not the time passed in
      const double averaged_time = horizon_coef_averager_.average_time(time);

      horizon_00 = averaged_horizon_coef_at_average_time.value()[0][0];
      dt_horizon_00 = averaged_horizon_coef_at_average_time.value()[1][0];
      const double d2t_horizon_00 =
          averaged_horizon_coef_at_average_time.value()[2][0];

      // Must account for time offset of averaged time. Do a simple Taylor
      // expansion
      const double time_diff = time - averaged_time;
      horizon_00 += time_diff * dt_horizon_00;
      dt_horizon_00 += 0.5 * square(time_diff) * d2t_horizon_00;

      // The "control error" for the averaged horizon coefficients is just the
      // averaged coefs minus the actual coef and time derivative from
      // apparent_horizon and time_deriv_apparent_horizon
      smoother_tuner_.update_timescale(
          std::array{
              DataVector{averaged_horizon_coef_at_average_time.value()[0][0]},
              DataVector{averaged_horizon_coef_at_average_time.value()[1][0]}} -
          std::array{
              DataVector{apparent_horizon.coefficients()[0]},
              DataVector{time_deriv_apparent_horizon.coefficients()[0]}});
    }

    // This is needed because the horizon_00 (and dt) are spherepack coefs, not
    // spherical harmonic coefs.
    const double spherepack_factor = sqrt(0.5 * M_PI);

    // This is needed for every state
    const double control_error_delta_r = size::control_error_delta_r(
        horizon_00, dt_horizon_00, lambda_00, dt_lambda_00,
        grid_frame_excision_sphere_radius);
    const std::optional<double> control_error_delta_r_outward =
        delta_r_drift_outward_options_.has_value()
            ? std::optional<double>(control_error_delta_r -
                                    delta_r_drift_outward_options_.value()
                                        .outward_drift_velocity -
                                    (lambda_00 +
                                     spherepack_factor * horizon_00 -
                                     grid_frame_excision_sphere_radius / Y00) /
                                        delta_r_drift_outward_options_.value()
                                            .outward_drift_timescale)
            : std::nullopt;

    info_.damping_time = min(tuner.current_timescale());

    const size::ErrorDiagnostics error_diagnostics = size::control_error(
        make_not_null(&info_), make_not_null(&char_speed_predictor_),
        make_not_null(&comoving_char_speed_predictor_),
        make_not_null(&delta_radius_predictor_), time, control_error_delta_r,
        control_error_delta_r_outward,
        delta_r_drift_outward_options_.has_value()
            ? std::optional<double>(delta_r_drift_outward_options_.value()
                                        .max_allowed_radial_distance)
            : std::nullopt,
        dt_lambda_00, apparent_horizon, excision_surface, lapse,
        shifty_quantity, spatial_metric_on_excision,
        inverse_spatial_metric_on_excision);

    state_history_.store(time, info_, error_diagnostics.control_error_args);

    if (Parallel::get<control_system::Tags::WriteDataToDisk>(cache)) {
      auto& observer_writer_proxy = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache);

      // \Delta R = < R_ah > - < R_ex >
      // < R_ah > = S_00 * Y_00
      // < R_ex > = R_ex^grid - \lambda_00 * Y_00
      // < \Delta R > = \Delta R / < R_ah >
      const double avg_delta_r =
          (spherepack_factor * horizon_00 + lambda_00) * Y00 -
          grid_frame_excision_sphere_radius;
      const double avg_relative_delta_r =
          avg_delta_r / (spherepack_factor * horizon_00 * Y00);

      Parallel::threaded_action<
          observers::ThreadedActions::WriteReductionDataRow>(
          observer_writer_proxy[0], subfile_name_, legend_,
          std::make_tuple(
              time, error_diagnostics.control_error,
              static_cast<double>(error_diagnostics.state_number),
              error_diagnostics.discontinuous_change_has_occurred ? 1.0 : 0.0,
              lambda_00, dt_lambda_00, horizon_00, dt_horizon_00,
              time_deriv_apparent_horizon.coefficients()[0],
              smoother_tuner_.current_timescale()[0],
              error_diagnostics.min_delta_r,
              error_diagnostics.min_relative_delta_r, avg_delta_r,
              avg_relative_delta_r,
              error_diagnostics.control_error_args.control_error_delta_r,
              error_diagnostics.target_char_speed,
              error_diagnostics.control_error_args.min_char_speed,
              error_diagnostics.min_comoving_char_speed,
              error_diagnostics.char_speed_crossing_time,
              error_diagnostics.comoving_char_speed_crossing_time,
              error_diagnostics.delta_r_crossing_time,
              error_diagnostics.suggested_timescale,
              error_diagnostics.damping_timescale));
    }

    if (Parallel::get<control_system::Tags::Verbosity>(cache) >=
        ::Verbosity::Verbose) {
      Parallel::printf("%s: %s\n", function_of_time_name,
                       error_diagnostics.update_message);
    }

    return DataVector{1, error_diagnostics.control_error};
  }

 private:
  TimescaleTuner<true> smoother_tuner_{};
  Averager<DerivOrder> horizon_coef_averager_{};
  size::Info info_{};
  intrp::ZeroCrossingPredictor char_speed_predictor_{};
  intrp::ZeroCrossingPredictor comoving_char_speed_predictor_{};
  intrp::ZeroCrossingPredictor delta_radius_predictor_{};
  size::StateHistory state_history_{};
  std::vector<std::string> legend_{};
  std::string subfile_name_{};
  std::optional<DeltaRDriftOutwardOptions> delta_r_drift_outward_options_{};
};
}  // namespace ControlErrors
}  // namespace control_system
