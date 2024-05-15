// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/IntegratedFunctionOfTime.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/ThreadsafeList.hpp"
#include "Domain/FunctionsOfTime/ThreadsafeList.tpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::FunctionsOfTime {

IntegratedFunctionOfTime::IntegratedFunctionOfTime() = default;
IntegratedFunctionOfTime::IntegratedFunctionOfTime(IntegratedFunctionOfTime&&) =
    default;
IntegratedFunctionOfTime::IntegratedFunctionOfTime(
    const IntegratedFunctionOfTime&) = default;
auto IntegratedFunctionOfTime::operator=(IntegratedFunctionOfTime&&)
    -> IntegratedFunctionOfTime& = default;
auto IntegratedFunctionOfTime::operator=(const IntegratedFunctionOfTime&)
    -> IntegratedFunctionOfTime& = default;
IntegratedFunctionOfTime::~IntegratedFunctionOfTime() = default;
IntegratedFunctionOfTime::IntegratedFunctionOfTime(
    CkMigrateMessage* /*unused*/) {}

IntegratedFunctionOfTime::IntegratedFunctionOfTime(
    const double t, std::array<double, 2> initial_func_and_derivs,
    const double expiration_time, const bool rotation)
    : deriv_info_at_update_times_(t), rotation_(rotation) {
  deriv_info_at_update_times_.insert(t, initial_func_and_derivs,
                                     expiration_time);
}

std::array<DataVector, 1> IntegratedFunctionOfTime::func(double t) const {
  return func_and_derivs<0>(t);
}
std::array<DataVector, 2> IntegratedFunctionOfTime::func_and_deriv(
    double t) const {
  return func_and_derivs<1>(t);
}
[[noreturn]] std::array<DataVector, 3>
IntegratedFunctionOfTime::func_and_2_derivs(double /*t*/) const {
  ERROR("Can only return first derivative.");
}

std::unique_ptr<FunctionOfTime> IntegratedFunctionOfTime::get_clone() const {
  return std::make_unique<IntegratedFunctionOfTime>(*this);
}

template <size_t MaxDerivReturned>
std::array<DataVector, MaxDerivReturned + 1>
IntegratedFunctionOfTime::func_and_derivs(const double t) const {
  static_assert(MaxDerivReturned <= 1, "can only return 1 derivative");

  const auto deriv_info_at_t = deriv_info_at_update_times_(t);
  std::array<DataVector, MaxDerivReturned + 1> result{};

  if (rotation_) {
    result.at(0) = DataVector{cos(deriv_info_at_t.data[0] / 2.), 0., 0.,
                              sin(deriv_info_at_t.data[0] / 2.)};
    if constexpr (MaxDerivReturned == 1) {
      result.at(1) = DataVector{
          -sin(deriv_info_at_t.data[0] / 2.) * deriv_info_at_t.data[1] / 2., 0.,
          0., cos(deriv_info_at_t.data[0] / 2.) * deriv_info_at_t.data[1] / 2.};
    }
  } else {
    for (size_t i = 0; i <= MaxDerivReturned; ++i) {
      result.at(i) = DataVector{gsl::at(deriv_info_at_t.data, i)};
    }
  }
  return result;
}

void IntegratedFunctionOfTime::update(const double time_of_update,
                                      DataVector updated_value_and_derivative,
                                      const double next_expiration_time) {
  if (time_of_update < deriv_info_at_update_times_.expiration_time()) {
    ERROR("Attempted to update from time "
          << time_of_update << " to time" << next_expiration_time
          << " but expiration time is "
          << deriv_info_at_update_times_.expiration_time());
  }

  ASSERT(updated_value_and_derivative.size() == 2,
         "The size of the DataVector should be 2: value and derivative");
  if (time_of_update != deriv_info_at_update_times_.expiration_time()) {
    update_backlog_[time_of_update] = std::make_pair(
        std::move(updated_value_and_derivative), next_expiration_time);
    return;
  }
  const std::array<double, 2> func{updated_value_and_derivative[0],
                                   updated_value_and_derivative[1]};
  deriv_info_at_update_times_.insert(time_of_update, func,
                                     next_expiration_time);
  while (not update_backlog_.empty() and
         update_backlog_.begin()->first == time_bounds()[1]) {
    const double current_exp_time = time_bounds()[1];
    const auto& item = update_backlog_[current_exp_time];
    const std::array<double, 2> updated_func{item.first[0], item.first[1]};
    deriv_info_at_update_times_.insert(current_exp_time, updated_func,
                                       item.second);
    update_backlog_.erase(current_exp_time);
  }
  deriv_info_at_update_times_.truncate_to_length(100);
}

std::array<double, 2> IntegratedFunctionOfTime::time_bounds() const {
  return {{deriv_info_at_update_times_.initial_time(),
           deriv_info_at_update_times_.expiration_time()}};
}

double IntegratedFunctionOfTime::expiration_after(const double time) const {
  return deriv_info_at_update_times_.expiration_after(time);
}

void IntegratedFunctionOfTime::pup(PUP::er& p) {
  FunctionOfTime::pup(p);
  size_t version = 0;
  p | version;
  if (version >= 0) {
    p | deriv_info_at_update_times_;
    p | rotation_;
    p | update_backlog_;
  }
}

bool operator==(const IntegratedFunctionOfTime& lhs,
                const IntegratedFunctionOfTime& rhs) {
  return lhs.deriv_info_at_update_times_ == rhs.deriv_info_at_update_times_ and
         lhs.rotation_ == rhs.rotation_ and
         lhs.update_backlog_ == rhs.update_backlog_;
}

bool operator!=(const IntegratedFunctionOfTime& lhs,
                const IntegratedFunctionOfTime& rhs) {
  return not(lhs == rhs);
}
PUP::able::PUP_ID IntegratedFunctionOfTime::my_PUP_ID = 0;  // NOLINT

#define DIMRETURNED(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<DataVector, DIMRETURNED(data) + 1>                     \
  IntegratedFunctionOfTime::func_and_derivs<DIMRETURNED(data)>(const double) \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1))

#undef DIMRETURNED
#undef INSTANTIATE
}  // namespace domain::FunctionsOfTime
