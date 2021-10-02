// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>

#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {

/// \brief Performs a barycentric interpolation with an order in a range fixed
/// at construction; this class can be chosen via the options factory mechanism
/// as a possible `SpanInterpolator`.
///
/// \details This will call a barycentric interpolation on a fixed minimum
/// length, so that buffers that adjust length based on
/// `required_points_before_and_after()` can be forced to use an interpolator of
/// a target order.
class BarycentricRationalSpanInterpolator : public SpanInterpolator {
 public:
  struct MinOrder {
    using type = size_t;
    static constexpr Options::String help = {
        "Order of barycentric interpolation"};
    static type lower_bound() { return 1; }
  };

  struct MaxOrder {
    using type = size_t;
    static constexpr Options::String help = {
        "Order of barycentric interpolation"};
    static type upper_bound() { return 10; }
  };

  using options = tmpl::list<MinOrder, MaxOrder>;
  static constexpr Options::String help = {
      "Barycentric interpolator of option-defined maximum and minimum order."};

  explicit BarycentricRationalSpanInterpolator(CkMigrateMessage* /*unused*/) {}

  WRAPPED_PUPable_decl_template(BarycentricRationalSpanInterpolator);  // NOLINT

  // clang-tidy: do not pass by non-const reference
  void pup(PUP::er& p) override {  // NOLINT
    p | min_order_;
    p | max_order_;
  }

  BarycentricRationalSpanInterpolator() = default;
  BarycentricRationalSpanInterpolator(
      const BarycentricRationalSpanInterpolator&) = default;
  BarycentricRationalSpanInterpolator& operator=(
      const BarycentricRationalSpanInterpolator&) = default;
  BarycentricRationalSpanInterpolator(BarycentricRationalSpanInterpolator&&) =
      default;
  BarycentricRationalSpanInterpolator& operator=(
      BarycentricRationalSpanInterpolator&&) = default;
  ~BarycentricRationalSpanInterpolator() override = default;

  explicit BarycentricRationalSpanInterpolator(size_t min_order,
                                               size_t max_order);

  std::unique_ptr<SpanInterpolator> get_clone() const override {
    return std::make_unique<BarycentricRationalSpanInterpolator>(*this);
  }

  // to get the generic overload:
  using SpanInterpolator::interpolate;

  double interpolate(const gsl::span<const double>& source_points,
                     const gsl::span<const double>& values,
                     double target_point) const override;

  size_t required_number_of_points_before_and_after() const override {
    return min_order_ / 2 + 1;
  }

 private:
  size_t min_order_ = 1;
  size_t max_order_ = 10;
};
}  // namespace intrp
