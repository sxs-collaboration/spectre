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

namespace control_system::ControlErrors {
template <::domain::ObjectLabel Horizon>
Size<Horizon>::Size(const int max_times) {
  const auto max_times_size_t = static_cast<size_t>(max_times);
  info_.state = std::make_unique<size::States::Initial>();
  char_speed_predictor_ = intrp::ZeroCrossingPredictor{3, max_times_size_t};
  comoving_char_speed_predictor_ =
      intrp::ZeroCrossingPredictor{3, max_times_size_t};
  delta_radius_predictor_ = intrp::ZeroCrossingPredictor{3, max_times_size_t};
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

template <::domain::ObjectLabel Horizon>
const std::optional<double>& Size<Horizon>::get_suggested_timescale() const {
  return info_.suggested_time_scale;
}

template <::domain::ObjectLabel Horizon>
bool Size<Horizon>::discontinuous_change_has_occurred() const {
  return info_.discontinuous_change_has_occurred;
}

template <::domain::ObjectLabel Horizon>
void Size<Horizon>::reset() {
  info_.reset();
}

template <::domain::ObjectLabel Horizon>
void Size<Horizon>::pup(PUP::er& p) {
  p | info_;
  p | char_speed_predictor_;
  p | comoving_char_speed_predictor_;
  p | delta_radius_predictor_;
  p | legend_;
  p | subfile_name_;
}

#define HORIZON(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template struct Size<HORIZON(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (::domain::ObjectLabel::A, ::domain::ObjectLabel::B,
                         ::domain::ObjectLabel::None))

#undef INSTANTIATE
#undef HORIZON
}  // namespace control_system::ControlErrors
