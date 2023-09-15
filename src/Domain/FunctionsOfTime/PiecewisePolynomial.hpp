// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTimeHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace domain {
namespace FunctionsOfTime {
/*!
 * \ingroup ComputationalDomainGroup
 * \brief A function that has a piecewise-constant `MaxDeriv`th derivative.
 *
 * \note This class conforms to the requirements of the
 * `Parallel::GlobalCache` for objects held by mutable global cache tags.
 */
template <size_t MaxDeriv>
class PiecewisePolynomial : public FunctionOfTime {
 public:
  PiecewisePolynomial() = default;
  PiecewisePolynomial(
      double t, std::array<DataVector, MaxDeriv + 1> initial_func_and_derivs,
      double expiration_time);

  ~PiecewisePolynomial() override = default;
  PiecewisePolynomial(PiecewisePolynomial&&);
  PiecewisePolynomial& operator=(PiecewisePolynomial&&);
  PiecewisePolynomial(const PiecewisePolynomial&);
  PiecewisePolynomial& operator=(const PiecewisePolynomial&);

  explicit PiecewisePolynomial(CkMigrateMessage* /*unused*/) {}

  auto get_clone() const -> std::unique_ptr<FunctionOfTime> override;

  // clang-tidy: google-runtime-references
  // clang-tidy: cppcoreguidelines-owning-memory,-warnings-as-errors
  WRAPPED_PUPable_decl_template(PiecewisePolynomial<MaxDeriv>);  // NOLINT

  /// Returns the function at an arbitrary time `t`.  If `MaxDeriv` is
  /// 0 and `update` has been called for time `t`, the updated value
  /// is ignored.
  std::array<DataVector, 1> func(double t) const override {
    return func_and_derivs<0>(t);
  }
  /// Returns the function and its first derivative at an arbitrary time `t`.
  /// If `MaxDeriv` is 1 and `update` has been called for time `t`, the updated
  /// value is ignored.
  std::array<DataVector, 2> func_and_deriv(double t) const override {
    return func_and_derivs<1>(t);
  }
  /// Returns the function and the first two derivatives at an arbitrary time
  /// `t`.  If `MaxDeriv` is 2 and `update` has been called for time `t`, the
  /// updated value is ignored.
  std::array<DataVector, 3> func_and_2_derivs(double t) const override {
    return func_and_derivs<2>(t);
  }

  /// Updates the `MaxDeriv`th derivative of the function at the given time.
  /// `updated_max_deriv` is a vector of the `MaxDeriv`ths for each component.
  /// `next_expiration_time` is the next expiration time.
  ///
  /// The \p time_of_update must be the same as the old expiration
  /// time.  It is passed as a check that the calling code is
  /// computing the other arguments with the correct value.
  void update(double time_of_update, DataVector updated_max_deriv,
              double next_expiration_time) override;

  /// Returns the domain of validity of the function,
  /// including the extrapolation region.
  std::array<double, 2> time_bounds() const override {
    return {{deriv_info_at_update_times_.front().time,
             expiration_time_.load(std::memory_order_acquire)}};
  }

  /// Return a const reference to the stored deriv info so external classes can
  /// read the stored times and derivatives (mostly for
  /// QuaternionFunctionOfTime).
  const std::list<FunctionOfTimeHelpers::StoredInfo<MaxDeriv + 1>>&
  get_deriv_info() const {
    return deriv_info_at_update_times_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  template <size_t LocalMaxDeriv>
  friend bool operator==(  // NOLINT(readability-redundant-declaration)
      const PiecewisePolynomial<LocalMaxDeriv>& lhs,
      const PiecewisePolynomial<LocalMaxDeriv>& rhs);

  template <size_t LocalMaxDeriv>
  friend std::ostream& operator<<(  // NOLINT(readability-redundant-declaration
      std::ostream& os,
      const PiecewisePolynomial<LocalMaxDeriv>& piecewise_polynomial);

  /// Returns the function and `MaxDerivReturned` derivatives at
  /// an arbitrary time `t`.
  /// The function has multiple components.
  template <size_t MaxDerivReturned = MaxDeriv>
  std::array<DataVector, MaxDerivReturned + 1> func_and_derivs(double t) const;

  // There exists a DataVector for each deriv order that contains
  // the values of that deriv order for all components.
  using value_type = std::array<DataVector, MaxDeriv + 1>;

  // In order for this class to be used in the mutable part of the global cache,
  // deriv_info_at_update_times_ must be a data type that preserves references
  // to elements upon insertion or resizing. A std::list fits this requirement
  std::list<FunctionOfTimeHelpers::StoredInfo<MaxDeriv + 1>>
      deriv_info_at_update_times_;
  alignas(64) std::atomic<double> expiration_time_{};
  // Pad memory to avoid false-sharing when accessing expiration_time_
  char unused_padding_expr_[64 - (sizeof(std::atomic<double>) % 64)] = {};
  alignas(64) std::atomic_uint64_t deriv_info_size_{};
  // Pad memory to avoid false-sharing when accessing deriv_info_size_
  char unused_padding_info_size_[64 - (sizeof(std::atomic_uint64_t) % 64)] = {};
  // No need to pup this since a default constructed mutex is always unlocked
  std::mutex update_mutex_{};
};

template <size_t MaxDeriv>
bool operator!=(const PiecewisePolynomial<MaxDeriv>& lhs,
                const PiecewisePolynomial<MaxDeriv>& rhs);

/// \cond
template <size_t MaxDeriv>
PUP::able::PUP_ID PiecewisePolynomial<MaxDeriv>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace FunctionsOfTime
}  // namespace domain
