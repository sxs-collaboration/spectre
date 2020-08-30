// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"

#include <pup.h>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep

// IWYU pragma: no_forward_declare Tensor

namespace Burgers::Solutions {

Step::Step(const double left_value, const double right_value,
           const double initial_shock_position, const OptionContext& context)
    : left_value_(left_value),
      right_value_(right_value),
      initial_shock_position_(initial_shock_position) {
  if (left_value <= right_value) {
    PARSE_ERROR(context, "Shock solution expects left_value > right_value");
  }
}

template <typename T>
Scalar<T> Step::u(const tnsr::I<T, 1>& x, const double t) const noexcept {
  const double current_shock_position =
      initial_shock_position_ + 0.5 * (left_value_ + right_value_) * t;
  return Scalar<T>(left_value_ -
                   (left_value_ - right_value_) *
                       step_function(get<0>(x) - current_shock_position));
}

tuples::TaggedTuple<Tags::U> Step::variables(const tnsr::I<DataVector, 1>& x,
                                             const double t,
                                             tmpl::list<Tags::U> /*meta*/) const
    noexcept {
  return {u(x, t)};
}

void Step::pup(PUP::er& p) noexcept {
  p | left_value_;
  p | right_value_;
  p | initial_shock_position_;
}

}  // namespace Burgers::Solutions

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                      \
  template Scalar<DTYPE(data)> Burgers::Solutions::Step::u(       \
      const tnsr::I<DTYPE(data), 1>& x, double t) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
