// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <vector>

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/Initial.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "NumericalAlgorithms/Interpolation/ZeroCrossingPredictor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"

namespace control_system {
namespace size {
double control_error_delta_r(const double horizon_00,
                             const double dt_horizon_00, const double lambda_00,
                             const double dt_lambda_00,
                             const double grid_frame_excision_sphere_radius) {
  const double Y00 = 0.25 * M_2_SQRTPI;

  // This corresponds to 'DeltaRPolicy=Relative' in SpEC.
  return dt_horizon_00 * (lambda_00 - grid_frame_excision_sphere_radius / Y00) /
             horizon_00 -
         dt_lambda_00;
}
}  // namespace size

namespace ControlErrors {
template <size_t DerivOrder, ::domain::ObjectLabel Horizon>
Size<DerivOrder, Horizon>::Size(const int max_times) {
  const auto max_times_size_t = static_cast<size_t>(max_times);
  info_.state = std::make_unique<size::States::Initial>();
  char_speed_predictor_ = intrp::ZeroCrossingPredictor{3, max_times_size_t};
  comoving_char_speed_predictor_ =
      intrp::ZeroCrossingPredictor{3, max_times_size_t};
  delta_radius_predictor_ = intrp::ZeroCrossingPredictor{3, max_times_size_t};
  state_history_ = size::StateHistory{DerivOrder + 1};
  legend_ = std::vector<std::string>{"Time",
                                     "ControlError",
                                     "StateNumber",
                                     "DiscontinuousChangeHasOccurred",
                                     "FunctionOfTime",
                                     "dtFunctionOfTime",
                                     "HorizonCoef00",
                                     "dtHorizonCoef00",
                                     "MinDeltaR",
                                     "MinRelativeDeltaR",
                                     "AvgDeltaR",
                                     "AvgRelativeDeltaR",
                                     "ControlErrorDeltaR",
                                     "TargetCharSpeed",
                                     "MinCharSpeed",
                                     "MinComovingCharSpeed",
                                     "CharSpeedCrossingTime",
                                     "ComovingCharSpeedCrossingTime",
                                     "DeltaRCrossingTime",
                                     "SuggestedTimescale",
                                     "DampingTime"};
  subfile_name_ = "/ControlSystems/Size" + get_output(Horizon) + "/Diagnostics";
}

template <size_t DerivOrder, ::domain::ObjectLabel Horizon>
const std::optional<double>&
Size<DerivOrder, Horizon>::get_suggested_timescale() const {
  return info_.suggested_time_scale;
}

template <size_t DerivOrder, ::domain::ObjectLabel Horizon>
bool Size<DerivOrder, Horizon>::discontinuous_change_has_occurred() const {
  return info_.discontinuous_change_has_occurred;
}

template <size_t DerivOrder, ::domain::ObjectLabel Horizon>
void Size<DerivOrder, Horizon>::reset() {
  info_.reset();
}

template <size_t DerivOrder, ::domain::ObjectLabel Horizon>
std::deque<std::pair<double, double>>
Size<DerivOrder, Horizon>::control_error_history() const {
  std::deque<std::pair<double, double>> history =
      state_history_.state_history(info_.state->number());
  // pop back so we don't include the current time, otherwise the averager
  // will error
  history.pop_back();
  return history;
}

template <size_t DerivOrder, ::domain::ObjectLabel Horizon>
void Size<DerivOrder, Horizon>::pup(PUP::er& p) {
  p | info_;
  p | char_speed_predictor_;
  p | comoving_char_speed_predictor_;
  p | delta_radius_predictor_;
  p | state_history_;
  p | legend_;
  p | subfile_name_;
}

#define DERIV_ORDER(data) BOOST_PP_TUPLE_ELEM(0, data)
#define HORIZON(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template struct Size<DERIV_ORDER(data), HORIZON(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3),
                        (::domain::ObjectLabel::A, ::domain::ObjectLabel::B,
                         ::domain::ObjectLabel::None))

#undef INSTANTIATE
#undef HORIZON
#undef DERIV_ORDER
}  // namespace ControlErrors
}  // namespace control_system
