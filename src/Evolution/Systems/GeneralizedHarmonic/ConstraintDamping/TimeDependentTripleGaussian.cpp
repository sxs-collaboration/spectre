// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TimeDependentTripleGaussian.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::ConstraintDamping {
TimeDependentTripleGaussian::TimeDependentTripleGaussian(CkMigrateMessage* msg)
    : DampingFunction<3, Frame::Grid>(msg) {}

TimeDependentTripleGaussian::TimeDependentTripleGaussian(
    const double constant, const double amplitude_1, const double width_1,
    const std::array<double, 3>& center_1, const double amplitude_2,
    const double width_2, const std::array<double, 3>& center_2,
    const double amplitude_3, const double width_3,
    const std::array<double, 3>& center_3,
    std::string function_of_time_for_scaling)
    : constant_(constant),
      amplitude_1_(amplitude_1),
      inverse_width_1_(1.0 / width_1),
      center_1_(center_1),
      amplitude_2_(amplitude_2),
      inverse_width_2_(1.0 / width_2),
      center_2_(center_2),
      amplitude_3_(amplitude_3),
      inverse_width_3_(1.0 / width_3),
      center_3_(center_3),
      function_of_time_for_scaling_(std::move(function_of_time_for_scaling)) {}

template <typename T>
void TimeDependentTripleGaussian::apply_call_operator(
    const gsl::not_null<Scalar<T>*> value_at_x,
    const tnsr::I<T, 3, Frame::Grid>& x, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.at(function_of_time_for_scaling_)
                 ->func(time)[0]
                 .size() == 1,
         "FunctionOfTimeForScaling in TimeDependentTripleGaussian must be a "
         "scalar FunctionOfTime, not "
             << functions_of_time.at(function_of_time_for_scaling_)
                    ->func(time)[0]
                    .size());
  const double function_of_time_value =
      functions_of_time.at(function_of_time_for_scaling_)->func(time)[0][0];

  // Start by setting the result to the constant
  get(*value_at_x) = constant_;

  // Loop over the three Gaussians, adding each to the result
  auto centered_coords = make_with_value<tnsr::I<T, 3, Frame::Grid>>(
      get<0>(x), std::numeric_limits<double>::signaling_NaN());

  const auto add_gauss_to_value_at_x =
      [&value_at_x, &centered_coords, &x, &function_of_time_value](
          const double amplitude, const double inverse_width,
          const std::array<double, 3>& center) {
        for (size_t i = 0; i < 3; ++i) {
          centered_coords.get(i) = x.get(i) - gsl::at(center, i);
        }
        get(*value_at_x) +=
            amplitude *
            exp(-get(dot_product(centered_coords, centered_coords)) *
                square(inverse_width * function_of_time_value));
      };
  add_gauss_to_value_at_x(amplitude_1_, inverse_width_1_, center_1_);
  add_gauss_to_value_at_x(amplitude_2_, inverse_width_2_, center_2_);
  add_gauss_to_value_at_x(amplitude_3_, inverse_width_3_, center_3_);
}  // namespace GeneralizedHarmonic::ConstraintDamping

void TimeDependentTripleGaussian::operator()(
    const gsl::not_null<Scalar<double>*> value_at_x,
    const tnsr::I<double, 3, Frame::Grid>& x, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  apply_call_operator(value_at_x, x, time, functions_of_time);
}
void TimeDependentTripleGaussian::operator()(
    const gsl::not_null<Scalar<DataVector>*> value_at_x,
    const tnsr::I<DataVector, 3, Frame::Grid>& x, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  destructive_resize_components(value_at_x, get<0>(x).size());
  apply_call_operator(value_at_x, x, time, functions_of_time);
}

void TimeDependentTripleGaussian::pup(PUP::er& p) {
  DampingFunction<3, Frame::Grid>::pup(p);
  p | constant_;
  p | amplitude_1_;
  p | inverse_width_1_;
  p | center_1_;
  p | amplitude_2_;
  p | inverse_width_2_;
  p | center_2_;
  p | amplitude_3_;
  p | inverse_width_3_;
  p | center_3_;
  p | function_of_time_for_scaling_;
}

auto TimeDependentTripleGaussian::get_clone() const
    -> std::unique_ptr<DampingFunction<3, Frame::Grid>> {
  return std::make_unique<TimeDependentTripleGaussian>(*this);
}

bool operator!=(const TimeDependentTripleGaussian& lhs,
                const TimeDependentTripleGaussian& rhs) {
  return not(lhs == rhs);
}
}  // namespace GeneralizedHarmonic::ConstraintDamping
PUP::able::PUP_ID GeneralizedHarmonic::ConstraintDamping::
    TimeDependentTripleGaussian::my_PUP_ID = 0;  // NOLINT
