// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace intrp {

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief A cubic B-spline interpolant of uniformly spaced samples
 *
 * Interpolates samples \f$f_i = f(t_0 + i \Delta t)\f$, \f$i = 0, \ldots,
 * N-1\f$, with a cubic B-spline. The interpolation error decreases as
 * \f$\mathcal{O}(\Delta t^4)\f$ for smooth data. This class wraps
 * `boost::math::interpolators::cardinal_cubic_b_spline` and adds:
 *
 * - Serialization: the samples, start time, and time step fully determine the
 *   interpolant, so it can be sent in a `pup` and rebuilt on the receiving
 *   side.
 * - Clamped evaluation: evaluation is clamped to the bounds of the sampled
 *   interval, so times that fall outside the interval by roundoff evaluate
 *   to the boundary values instead of extrapolating the spline. Times
 *   outside the bounds beyond roundoff trigger an `ASSERT` in debug builds.
 *
 * Here is an example how to use this class:
 *
 * \snippet Test_UniformCardinalBSpline.cpp uniform_cardinal_b_spline_example
 *
 * \note Requires Boost 1.81 or newer at runtime because of an accuracy fix
 * for the estimate of the derivative at the right endpoint in the Boost
 * implementation, see
 * https://github.com/boostorg/math/commit/4809e714d4806c07da3a3def0c4550daa0529b8d.
 * The constructor raises an error for older Boost versions.
 */
class UniformCardinalBSpline {
 public:
  /*!
   * \brief Construct from uniformly spaced samples.
   *
   * \param values The sampled function values \f$f(t_0 + i \Delta t)\f$. At
   *     least 5 samples are required.
   * \param start_time The time \f$t_0\f$ of the first sample.
   * \param time_step The (positive) spacing \f$\Delta t\f$ between samples.
   */
  UniformCardinalBSpline(std::vector<double> values, double start_time,
                         double time_step);

  UniformCardinalBSpline() = default;

  /// Evaluate the interpolant at `time`, clamped to the `bounds()`
  double operator()(double time) const;

  /// The sampled function values
  const std::vector<double>& values() const { return values_; }

  /// The time of the first sample
  double start_time() const { return start_time_; }

  /// The spacing between samples
  double time_step() const { return time_step_; }

  /// The first and last sample time
  std::array<double, 2> bounds() const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  void initialize_interpolant();

  std::vector<double> values_{};
  double start_time_{};
  double time_step_{};
  std::optional<boost::math::interpolators::cardinal_cubic_b_spline<double>>
      interpolant_{};
};

bool operator==(const UniformCardinalBSpline& lhs,
                const UniformCardinalBSpline& rhs);

bool operator!=(const UniformCardinalBSpline& lhs,
                const UniformCardinalBSpline& rhs);

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Estimate the maximum interpolation error of a
 * `intrp::UniformCardinalBSpline` through the given samples.
 *
 * Measures the deviation of an interpolant through every other sample from
 * the sampled values at the sample times, where the error of that coarser
 * interpolant peaks. For smooth data the interpolation error of a cubic
 * spline decreases by a factor of 16 when the step size is halved, so the
 * error of the interpolant through all samples is about 1/16 of the measured
 * deviation for smooth data in the convergent regime. The deviation is divided
 * by only 8 to obtain a conservative estimate, accounting for nonsmooth data
 * that converges slower.
 *
 * \param values The sampled function values. At least 9 samples are required
 *     so the coarser interpolant has at least 5.
 * \param start_time The time of the first sample.
 * \param time_step The (positive) spacing between samples.
 */
double estimate_interpolation_error(const std::vector<double>& values,
                                    double start_time, double time_step);

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compress uniformly spaced samples into a
 * `intrp::UniformCardinalBSpline` with fewer points that reproduces the
 * samples to the given tolerance.
 *
 * Builds a reference interpolant through all samples and resamples it on
 * coarser uniform grids over the same time interval, starting with 6 points
 * and doubling until the resampled interpolant reproduces the input samples
 * to within `absolute_tolerance` at the sample times. This reduces memory
 * for smooth, densely sampled data, e.g. time series of spectral modes in a
 * simulation.
 *
 * \returns The compressed interpolant and its maximum absolute deviation
 * from the input samples at the sample times.
 *
 * If the tolerance cannot be met with fewer points than the input, the
 * returned interpolant holds the original samples. Its deviation from the
 * samples at the sample times vanishes by construction, so the returned
 * error is the error of the last (coarser) candidate in the doubling search
 * as a conservative estimate, and may exceed the tolerance. Inputs with 6 or
 * fewer samples are returned unchanged with zero error.
 *
 * \param values The sampled function values. At least 5 samples are required.
 * \param start_time The time of the first sample.
 * \param time_step The (positive) spacing between samples.
 * \param absolute_tolerance Maximum allowed absolute deviation of the
 *     compressed interpolant from the input samples.
 */
std::pair<UniformCardinalBSpline, double> compress_to_tolerance(
    const std::vector<double>& values, double start_time, double time_step,
    double absolute_tolerance);

}  // namespace intrp
