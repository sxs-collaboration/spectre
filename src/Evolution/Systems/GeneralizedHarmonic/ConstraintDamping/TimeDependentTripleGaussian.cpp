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
#include "Options/ParseError.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace gh::ConstraintDamping {
TimeDependentTripleGaussian::TimeDependentTripleGaussian(CkMigrateMessage* msg)
    : DampingFunction<3, Frame::Grid>(msg) {}

TimeDependentTripleGaussian::TimeDependentTripleGaussian(
    const double constant, const double amplitude_1, const double width_1,
    const std::optional<std::array<double, 3>>& center_1,
    const double amplitude_2, const double width_2,
    const std::optional<std::array<double, 3>>& center_2,
    const double amplitude_3, const double width_3,
    const std::array<double, 3>& center_3, const std::string& movement_method,
    const Options::Context& context)
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
      movement_method_(movement_method == "ExpansionFactor"
                           ? MovementMethods::ExpansionFactor
                           : MovementMethods::ObjectCenters) {
  if (movement_method != "ExpansionFactor" and
      movement_method != "ObjectCenters") {
    PARSE_ERROR(
        context,
        "The movement method must be either 'ExpansionFactor' (for BBH "
        "simulations) or 'ObjectCenters' (for BNS simulations) but got '"
            << movement_method << "'");
  }
  if (movement_method == "ObjectCenters") {
    if (center_1_.has_value()) {
      PARSE_ERROR(context,
                  "You cannot set the Center of Gaussian1 when using the "
                  "ObjectCenters movement method. The center is determine from "
                  "the ObjectCenters function of time.");
    }
    if (center_2_.has_value()) {
      PARSE_ERROR(context,
                  "You cannot set the Center of Gaussian2 when using the "
                  "ObjectCenters movement method. The center is determine from "
                  "the ObjectCenters function of time.");
    }
  } else {
    if (not center_1_.has_value()) {
      PARSE_ERROR(context,
                  "You must set the Center of Gaussian1 when using the "
                  "ExpansionFactor movement method.");
    }
    if (not center_2_.has_value()) {
      PARSE_ERROR(context,
                  "You must set the Center of Gaussian1 when using the "
                  "ExpansionFactor movement method.");
    }
  }
}

template <typename T>
void TimeDependentTripleGaussian::apply_call_operator(
    const gsl::not_null<Scalar<T>*> value_at_x,
    const tnsr::I<T, 3, Frame::Grid>& x, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  // Start by setting the result to the constant
  get(*value_at_x) = constant_;

  // Loop over the three Gaussians, adding each to the result
  auto centered_coords = make_with_value<tnsr::I<T, 3, Frame::Grid>>(
      get<0>(x), std::numeric_limits<double>::signaling_NaN());

  const auto add_gauss_to_value_at_x =
      [&centered_coords, &x, &functions_of_time, time, this, &value_at_x](
          const double amplitude, const double inverse_width,
          const std::array<double, 3>& center) {
        for (size_t i = 0; i < 3; ++i) {
          centered_coords.get(i) = x.get(i) - gsl::at(center, i);
        }
        if (this->movement_method_ == MovementMethods::ExpansionFactor) {
          ASSERT(functions_of_time.at(function_of_time_for_scaling_)
                         ->func(time)[0]
                         .size() == 1,
                 "FunctionOfTimeForScaling in TimeDependentTripleGaussian must "
                 "be a scalar FunctionOfTime, not "
                     << functions_of_time.at(function_of_time_for_scaling_)
                            ->func(time)[0]
                            .size());
          const double expansion_factor_value =
              functions_of_time.at(function_of_time_for_scaling_)
                  ->func(time)[0][0];
          get(*value_at_x) +=
              amplitude *
              exp(-get(dot_product(centered_coords, centered_coords)) *
                  square(inverse_width * expansion_factor_value));
        } else {
          get(*value_at_x) +=
              amplitude *
              exp(-get(dot_product(centered_coords, centered_coords)) *
                  square(inverse_width));
        }
      };
  if (this->movement_method_ == MovementMethods::ExpansionFactor) {
    add_gauss_to_value_at_x(amplitude_1_, inverse_width_1_, center_1_.value());
    add_gauss_to_value_at_x(amplitude_2_, inverse_width_2_, center_2_.value());
  } else {
    const DataVector centers =
        functions_of_time.at(function_of_time_for_centers_)->func(time)[0];
    ASSERT(centers.size() == 6,
           "FunctionOfTimeForCenters in TimeDependentTripleGaussian must have "
           "6 components, not "
               << functions_of_time.at(function_of_time_for_centers_)
                      ->func(time)[0]
                      .size());
    const std::array center_1{centers[0], centers[1], centers[2]};
    const std::array center_2{centers[3], centers[4], centers[5]};
    add_gauss_to_value_at_x(amplitude_1_, inverse_width_1_, center_1);
    add_gauss_to_value_at_x(amplitude_2_, inverse_width_2_, center_2);
  }
  // Gaussian 3 should be the one centered at the origin in a binary simulation.
  add_gauss_to_value_at_x(amplitude_3_, inverse_width_3_, center_3_);
}  // namespace gh::ConstraintDamping

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
  set_number_of_grid_points(value_at_x, x);
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
  p | movement_method_;
}

auto TimeDependentTripleGaussian::get_clone() const
    -> std::unique_ptr<DampingFunction<3, Frame::Grid>> {
  return std::make_unique<TimeDependentTripleGaussian>(*this);
}

bool operator==(const TimeDependentTripleGaussian& lhs,
                const TimeDependentTripleGaussian& rhs) {
  return lhs.constant_ == rhs.constant_ and
         lhs.amplitude_1_ == rhs.amplitude_1_ and
         lhs.inverse_width_1_ == rhs.inverse_width_1_ and
         lhs.center_1_ == rhs.center_1_ and
         lhs.amplitude_2_ == rhs.amplitude_2_ and
         lhs.inverse_width_2_ == rhs.inverse_width_2_ and
         lhs.center_2_ == rhs.center_2_ and
         lhs.amplitude_3_ == rhs.amplitude_3_ and
         lhs.inverse_width_3_ == rhs.inverse_width_3_ and
         lhs.center_3_ == rhs.center_3_ and
         lhs.movement_method_ == rhs.movement_method_;
}

bool operator!=(const TimeDependentTripleGaussian& lhs,
                const TimeDependentTripleGaussian& rhs) {
  return not(lhs == rhs);
}
}  // namespace gh::ConstraintDamping
PUP::able::PUP_ID
    gh::ConstraintDamping::TimeDependentTripleGaussian::my_PUP_ID =
        0;  // NOLINT
