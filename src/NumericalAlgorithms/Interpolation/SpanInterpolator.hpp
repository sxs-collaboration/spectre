// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {

class LinearSpanInterpolator;
class CubicSpanInterpolator;
class BarycentricRationalSpanInterpolator;

/// \brief Base class for interpolators so that the factory options
/// mechanism can be used.
///
/// \details The virtual functions in this class demand only that the real
/// `interpolate` function, `get_clone` function, and
/// `required_number_of_points_before_and_after` function be overridden in the
/// derived class. The `interpolate` for complex values can just be used from
/// this base class, which calls the real version for each component. If it is
/// possible to make a specialized complex version that avoids allocations, that
/// is probably more efficient.
class SpanInterpolator : public PUP::able {
 public:
  using creatable_classes =
      tmpl::list<LinearSpanInterpolator, CubicSpanInterpolator,
                 BarycentricRationalSpanInterpolator>;

  WRAPPED_PUPable_abstract(SpanInterpolator);  // NOLINT

  /// Produce a `std::unique_ptr` that points to a copy of `*this``
  virtual std::unique_ptr<SpanInterpolator> get_clone() const = 0;

  /// Perform the interpolation of function represented by `values` at
  /// `source_points` to the requested `target_point`, returning the
  /// interpolation result.
  virtual double interpolate(const gsl::span<const double>& source_points,
                             const gsl::span<const double>& values,
                             double target_point) const = 0;

  /// Perform the interpolation of function represented by complex `values` at
  /// `source_points` to the requested `target_point`, returning the
  /// (complex) interpolation result.
  std::complex<double> interpolate(
      const gsl::span<const double>& source_points,
      const gsl::span<const std::complex<double>>& values,
      double target_point) const;

  /// The number of domain points that should be both before and after the
  /// requested target point for best interpolation. For instance, for a linear
  /// interpolator, this function would return `1` to request that the target is
  /// between the two domain points passed to `source_points`.
  virtual size_t required_number_of_points_before_and_after() const = 0;
};
}  // namespace intrp
