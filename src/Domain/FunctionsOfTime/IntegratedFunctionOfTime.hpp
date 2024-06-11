// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/ThreadsafeList.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::FunctionsOfTime {
/*!
 * \brief A function that is integrated manually.
 * \details This function only works with global time steppers that have
 * strictly positively increasing substeps. When evaluated at this global time
 * step, it returns the function and first derivative at that time step. It
 * therefore needs to be updated every time step with its current values.
 */
class IntegratedFunctionOfTime : public FunctionOfTime {
 public:
  IntegratedFunctionOfTime();
  IntegratedFunctionOfTime(IntegratedFunctionOfTime&&);
  IntegratedFunctionOfTime(const IntegratedFunctionOfTime&);
  IntegratedFunctionOfTime& operator=(IntegratedFunctionOfTime&&);
  IntegratedFunctionOfTime& operator=(const IntegratedFunctionOfTime&);
  ~IntegratedFunctionOfTime() override;
  /*!
   * \brief Constructs the function using the initial time, initial values,
   * derivative and expiration time. If `rotation` is true, the function will be
   * converted to a quaternion before it is output as used by the `Rotation`
   * map.
   */
  IntegratedFunctionOfTime(double t,
                           std::array<double, 2> initial_func_and_derivs,
                           double expiration_time, bool rotation);

  // clang-tidy: google-runtime-references
  // clang-tidy: cppcoreguidelines-owning-memory,-warnings-as-errors
  WRAPPED_PUPable_decl_template(IntegratedFunctionOfTime);  // NOLINT

  explicit IntegratedFunctionOfTime(CkMigrateMessage* /*unused*/);

  auto get_clone() const -> std::unique_ptr<FunctionOfTime> override;

  std::array<DataVector, 1> func(double t) const override;
  std::array<DataVector, 2> func_and_deriv(double t) const override;
  [[noreturn]] std::array<DataVector, 3> func_and_2_derivs(
      double /*t*/) const override;

  /*!
   * \brief Updates the function to the next global time step. The
   * `updated_value_and_derivative` argument needs to be a DataVector of size 2,
   * with the zeroth element holding the function's value and the first element
   * holding its derivative. If `rotation_` is set to true, this corresponds to
   * the angle and the angular velocity for a rotation about the z-axis.
   */
  void update(double time_of_update, DataVector updated_value_and_derivative,
              double next_expiration_time) override;

  double expiration_after(double time) const override;

  std::array<double, 2> time_bounds() const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(  // NOLINT(readability-redundant-declaration)
      const IntegratedFunctionOfTime& lhs, const IntegratedFunctionOfTime& rhs);

  template <size_t MaxDerivReturned>
  std::array<DataVector, MaxDerivReturned + 1> func_and_derivs(double t) const;
  FunctionOfTimeHelpers::ThreadsafeList<std::array<double, 2>>
      deriv_info_at_update_times_;
  std::map<double, std::pair<DataVector, double>> update_backlog_{};
  bool rotation_ = false;
};

bool operator!=(const IntegratedFunctionOfTime& lhs,
                const IntegratedFunctionOfTime& rhs);
}  // namespace domain::FunctionsOfTime
