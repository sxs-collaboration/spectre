// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {

/// \brief Performs a cubic interpolation; this class can be chosen via the
/// options factory mechanism as a possible `SpanInterpolator`.
///
/// \details This interpolator is hand-coded to be identical to the SpEC
/// implementation used for SpEC CCE so that comparison results can be as close
/// as possible for diagnostics.
class CubicSpanInterpolator : public SpanInterpolator {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {"Cubic interpolator."};

  CubicSpanInterpolator() = default;
  CubicSpanInterpolator(const CubicSpanInterpolator&) = default;
  CubicSpanInterpolator& operator=(const CubicSpanInterpolator&) = default;
  CubicSpanInterpolator(CubicSpanInterpolator&&) = default;
  CubicSpanInterpolator& operator=(CubicSpanInterpolator&&) = default;
  ~CubicSpanInterpolator() override = default;

  explicit CubicSpanInterpolator(CkMigrateMessage* /*unused*/) {}

  WRAPPED_PUPable_decl_template(CubicSpanInterpolator);  // NOLINT

  // clang-tidy: do not pass by non-const reference
  void pup(PUP::er& /*p*/) override {}

  std::unique_ptr<SpanInterpolator> get_clone() const override {
    return std::make_unique<CubicSpanInterpolator>(*this);
  }

  double interpolate(const gsl::span<const double>& source_points,
                     const gsl::span<const double>& values,
                     double target_point) const override;

  std::complex<double> interpolate(
      const gsl::span<const double>& source_points,
      const gsl::span<const std::complex<double>>& values,
      double target_point) const;

  size_t required_number_of_points_before_and_after() const override {
    return 2;
  }
};
}  // namespace intrp
