// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <optional>
#include <pup.h>
#include <string>
#include <utility>

#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/Tags/ObjectCenter.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Algorithm.hpp"
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
 * \brief Control error in the for the
 * `domain::CoordinateMaps::TimeDependent::Skew` map.
 *
 * \details Computes the error in the map parameters $F_y(t)$ and $F_z(t)$ of
 * the \link domain::CoordinateMaps::TimeDependent::Skew Skew \endlink using
 * a modified version of Eq. (71) from \cite Hemberger2012jz. This modified
 * control error is
 *
 * \begin{equation}
 * Q_{F_j} = g\left(\frac{w_A \Theta_A^j + w_B \Theta_B^j}{w_A + w_B}\right)
 *     - (1-g)F_j
 * \end{equation}
 *
 * where the falloff function is assumed to be $W(\vec{x}) \approx 1$ from the
 * \link domain::CoordinateMaps::TimeDependent::Skew Skew \endlink map,
 * $\Theta_H^j$ are the inclination angles between the $x$-axis and the normal
 * to the horizon at the intersection point $x_H^\textrm{Int}$ between the
 * $x$-axis and the horizon, $w_A$ and $w_B$ are averaging weights defined by
 *
 * \begin{equation}
 * w_H = \exp{\left(-\frac{x^0_C - x_H^\textrm{Int}}{x^0_C - C^0_H}\right)}
 * \end{equation}
 *
 * with $C^0_H$ being the centers of the excision boundaries, and finally with
 *
 * \begin{equation}
 * g = \frac{1}{2}\left(1-\tanh\left(10\frac{x_A^\textrm{Int} -
 * x_B^\textrm{Int}}{C^0_A - C^0_B} - 5\right)\right).
 * \end{equation}
 *
 * This transition function $g$ is meant to only activate skew control when the
 * black holes are close to merger to avoid adverse effects with junk radiation.
 * If $g < 0.0025$, then we turn skew control off completely and the control
 * error just becomes
 *
 * \begin{equation}
 * Q_{F_j} = -F_j.
 * \end{equation}
 *
 * This threshold value of $g = 0.0025$ was chosen in SpEC and seems to work
 * well.
 *
 * Requirements:
 * - This control error requires that there be exactly two objects in the
 *   simulation
 * - Currently both these objects must be black holes
 * - Currently this control system can only be used with the \link
 *   control_system::Systems::Skew Skew \endlink control system
 */
struct Skew : tt::ConformsTo<protocols::ControlError> {
  using object_centers = domain::object_list<>;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Computes the control error for skew control."};

  // Explicitly defined copy constructor because DataBox isn't copyable
  Skew() = default;
  Skew(Skew&& rhs) = default;
  Skew& operator=(Skew&& rhs) = default;
  Skew(const Skew&);
  Skew& operator=(const Skew&);
  ~Skew() = default;

  /*!
   * \brief Returns the internal suggested timescale. A std::nullopt means that
   * no timescale is suggested.
   */
  std::optional<double> get_suggested_timescale() const;

  /*!
   * \brief Resets the internal suggested timescale to nullopt.
   */
  void reset();

  void pup(PUP::er& p);

  template <typename Metavariables, typename... TupleTags>
  DataVector operator()(const ::TimescaleTuner<true>& /*tuner*/,
                        const Parallel::GlobalCache<Metavariables>& cache,
                        const double time,
                        const std::string& function_of_time_name,
                        const tuples::TaggedTuple<TupleTags...>& measurements) {
    const ylm::Strahlkorper<Frame::Distorted>& horizon_a = tuples::get<
        QueueTags::Horizon<Frame::Distorted, ::domain::ObjectLabel::A>>(
        measurements);
    const ylm::Strahlkorper<Frame::Distorted>& horizon_b = tuples::get<
        QueueTags::Horizon<Frame::Distorted, ::domain::ObjectLabel::B>>(
        measurements);

    // Copy the horizon into the box so the compute tags use the correct
    // strahlkorper
    const auto set_horizon =
        [](const gsl::not_null<ylm::Strahlkorper<Frame::Distorted>*>
               horizon_ptr,
           const ylm::Strahlkorper<Frame::Distorted>& horizon) {
          *horizon_ptr = horizon;
        };

    db::mutate<ylm::Tags::Strahlkorper<Frame::Distorted>>(
        set_horizon, make_not_null(&box_a_), horizon_a);
    db::mutate<ylm::Tags::Strahlkorper<Frame::Distorted>>(
        set_horizon, make_not_null(&box_b_), horizon_b);

    const auto& normal_one_form_a =
        db::get<ylm::Tags::NormalOneForm<Frame::Distorted>>(box_a_);
    const auto& cartesian_coords_a =
        db::get<ylm::Tags::CartesianCoords<Frame::Distorted>>(box_a_);
    const auto& center_a =
        Parallel::get<domain::Tags::ObjectCenter<domain::ObjectLabel::A>>(
            cache);

    const auto& normal_one_form_b =
        db::get<ylm::Tags::NormalOneForm<Frame::Distorted>>(box_b_);
    const auto& cartesian_coords_b =
        db::get<ylm::Tags::CartesianCoords<Frame::Distorted>>(box_b_);
    const auto& center_b =
        Parallel::get<domain::Tags::ObjectCenter<domain::ObjectLabel::B>>(
            cache);

    // The translation of the cutting plane in the grid frame is just the
    // average of the two centers.
    const double cut_x = 0.5 * (center_a[0] + center_b[0]);

    // For AhA since it always has x_center > cut_x, we want the point which is
    // closest to the cutting plane, which in terms of theta, phi = pi/2, pi.
    // For AhB since it always has x_center < cut_x, we want the point which is
    // closest to the cutting plane, which in terms of theta, phi = pi/2, 0.
    // To get the coord and normal one form at these theta/phi, we do an
    // interpolation for both Ah's
    const auto& ylm_a = horizon_a.ylm_spherepack();
    const ylm::Spherepack::InterpolationInfo<double> interpolation_info_a =
        ylm_a.set_up_interpolation_info(std::array{M_PI_2, M_PI});
    const auto& ylm_b = horizon_b.ylm_spherepack();
    const ylm::Spherepack::InterpolationInfo<double> interpolation_info_b =
        ylm_b.set_up_interpolation_info(std::array{M_PI_2, 0.0});

    const auto set_intersection =
        [](const gsl::not_null<DataVector*> inclination_angle,
           const gsl::not_null<std::array<double, 3>*> intersection_coord,
           const ylm::Spherepack& ylm,
           const ylm::Spherepack::InterpolationInfo<double>& interpolation_info,
           const auto& normal_one_form, const auto& cartesian_coords) {
          std::array<double, 3> intersection_normal_one_form{};
          for (size_t i = 0; i < 3; i++) {
            ylm.interpolate(make_not_null(&gsl::at(*intersection_coord, i)),
                            make_not_null(cartesian_coords.get(i).data()),
                            interpolation_info);
            ylm.interpolate(
                make_not_null(&gsl::at(intersection_normal_one_form, i)),
                make_not_null(normal_one_form.get(i).data()),
                interpolation_info);
          }

          for (size_t i = 0; i < 2; ++i) {
            (*inclination_angle)[i] =
                atan2(gsl::at(intersection_normal_one_form, i + 1),
                      (intersection_normal_one_form)[0]);
            if ((*inclination_angle)[i] < -0.5 * M_PI) {
              (*inclination_angle)[i] += M_PI;
            } else if ((*inclination_angle)[i] > +0.5 * M_PI) {
              (*inclination_angle)[i] -= M_PI;
            }
          }
        };

    // These are the values we need at the theta/phi described above
    std::array<double, 3> intersection_coord_a{};
    std::array<double, 3> intersection_coord_b{};

    set_intersection(make_not_null(&inclination_angle_a_),
                     make_not_null(&intersection_coord_a), ylm_a,
                     interpolation_info_a, normal_one_form_a,
                     cartesian_coords_a);
    set_intersection(make_not_null(&inclination_angle_b_),
                     make_not_null(&intersection_coord_b), ylm_b,
                     interpolation_info_b, normal_one_form_b,
                     cartesian_coords_b);

    const double relative_delta_x =
        (intersection_coord_a[0] - intersection_coord_b[0]) /
        (center_a[0] - center_b[0]);
    // Hardcoded function used in SpEC
    const double temporal_transition_function =
        0.5 * (1.0 - tanh(10.0 * relative_delta_x - 5.0));

    // Hardcoded value used in SpEC
    constexpr double activation_threshold = 0.0025;

    const auto& function_of_time =
        Parallel::get<domain::Tags::FunctionsOfTime>(cache).at(
            function_of_time_name);

    // Only activate if BHs are close enough. Otherwise control to zero
    DataVector func = std::move(function_of_time->func(time)[0]);
    DataVector& control_error = func;
    if (temporal_transition_function > activation_threshold) {
      suggested_timescale_ =
          std::max(std::min(abs(center_a[0]), abs(center_b[0])),
                   std::min(abs(cut_x - intersection_coord_a[0]),
                            abs(cut_x - intersection_coord_b[0])));

      const double weight_a =
          exp(-(cut_x - intersection_coord_a[0]) / (cut_x - center_a[0]));
      const double weight_b =
          exp(-(cut_x - intersection_coord_b[0]) / (cut_x - center_b[0]));

      control_error = temporal_transition_function *
                          (weight_a * inclination_angle_a_ +
                           weight_b * inclination_angle_b_) /
                          (weight_a + weight_b) -
                      func;
    } else {
      control_error *= -1.0;
    }

    return std::move(control_error);
  }

 private:
  std::optional<double> suggested_timescale_;
  // not pupped. These are allocated here to avoid memory allocations during
  // the operator() call. All their data is temporary
  DataVector inclination_angle_a_{2, 0.0};
  DataVector inclination_angle_b_{2, 0.0};
  db::compute_databox_type<
      tmpl::append<ylm::Tags::items_tags<Frame::Distorted>,
                   ylm::Tags::compute_items_tags<Frame::Distorted>>>
      box_a_;
  db::compute_databox_type<
      tmpl::append<ylm::Tags::items_tags<Frame::Distorted>,
                   ylm::Tags::compute_items_tags<Frame::Distorted>>>
      box_b_;
};
}  // namespace control_system::ControlErrors
