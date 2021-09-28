// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class ComplexDataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Spectral {
namespace Swsh {

/// \ingroup SwshGroup
/// A utility for evaluating a particular spin-weighted spherical harmonic
/// function at arbitrary points.
///
/// \warning This should NOT be used for interpolation; such an evaluation
/// strategy is hopelessly slow compared to Clenshaw recurrence strategies.
/// Instead use `SwshInterpolator`.
class SpinWeightedSphericalHarmonic {
 public:
  // charm needs an empty constructor
  SpinWeightedSphericalHarmonic() = default;

  SpinWeightedSphericalHarmonic(int spin, size_t l, int m);

  /*!
   *  Return by pointer the values of the spin-weighted spherical harmonic
   *  evaluated at `theta` and `phi`.
   *
   *  \details The additional values `sin_theta_over_2` and `cos_theta_over_2`,
   *  representing \f$\sin(\theta/2)\f$ and \f$\cos(\theta/2)\f$ are taken as
   *  required input to improve the speed of evaluation when called more than
   *  once.
   *
   *  The formula we evaluate (with various prefactors precomputed, cached, and
   *  optimized from the factorials) is \cite Goldberg1966uu
   *
   *  \f{align*}{
   *  {}_s Y_{l m} = (-1)^m \sqrt{\frac{(l + m)! (l-m)! (2l + 1)}
   *  {4 \pi (l + s)! (l - s)!}} \sin^{2 l}(\theta / 2)
   *   \sum_{r = 0}^{l - s} {l - s \choose r} {l + s \choose r + s - m}
   *  (-1)^{l - r - s} e^{i m \phi} \cot^{2 r + s - m}(\theta / 2).
   *  \f}
   */
  void evaluate(gsl::not_null<ComplexDataVector*> result,
                const DataVector& theta, const DataVector& phi,
                const DataVector& sin_theta_over_2,
                const DataVector& cos_theta_over_2) const;

  /// Return by value the spin-weighted spherical harmonic evaluated at `theta`
  /// and `phi`.
  ///
  /// \details The additional values `sin_theta_over_2` and `cos_theta_over_2`,
  /// representing \f$\sin(\theta/2)\f$ and \f$\cos(\theta/2)\f$ are taken as
  /// required input to improve the speed of evaluation when called more than
  /// once.
  ComplexDataVector evaluate(const DataVector& theta, const DataVector& phi,
                             const DataVector& sin_theta_over_2,
                             const DataVector& cos_theta_over_2) const;

  /// Return by value the spin-weighted spherical harmonic evaluated at `theta`
  /// and `phi`.
  std::complex<double> evaluate(double theta, double phi) const;

  /// Serialization for Charm++.
  void pup(PUP::er& p);  // NOLINT

 private:
  int spin_ = 0;
  size_t l_ = 0;
  int m_ = 0;
  double overall_prefactor_ = 0.0;
  std::vector<double> r_prefactors_;
};

/*!
 * \ingroup SwshGroup
 * \brief Performs interpolation for spin-weighted spherical harmonics by
 * taking advantage of the Clenshaw method of expanding recurrence relations.
 *
 * \details During construction, we cache several functions of the target
 * interpolation points that will be used during the Clenshaw evaluation.
 * A new `SwshInterpolator` object must be created for each new set of target
 * points, but the member function `SwshInterpolator::interpolate()` may be
 * called on several different coefficients or collocation sets, and of
 * different spin-weights.
 *
 * Recurrence constants
 * --------------------
 * This utility obtains the Clenshaw interpolation constants from a
 * `StaticCache`, so that 'universal' quantities can be calculated only once per
 * execution and re-used on each interpolation.
 *
 * We evaluate the recurrence coefficients \f$\alpha_l^{(a,b)}\f$ and
 * \f$\beta_l^{(a,b)}\f$, where \f$a = |s + m|\f$, \f$b = |s - m|\f$, and
 *
 * \f[
 * {}_sY_{l, m}(\theta, \phi) = \alpha_l^{(a,b)}(\theta) {}_s Y_{l-1, m}
 * +  \beta_l^{(a, b)} {}_s Y_{l - 2, m}(\theta, \phi)
 * \f]
 *
 * The core Clenshaw recurrence is in the\f$l\f$ modes of the
 * spin-weighted spherical harmonic. For a set of modes \f$a_l^{(a,b)}\f$ at
 * fixed \f$m\f$, the function value is evaluated by the recurrence:
 *
 * \f{align*}{
 * y^{(a, b)}_{l_\text{max} + 2} &= y^{(a, b)}_{l_\text{max} + 1} = 0 \\
 * y^{(a, b)}_{l}(\theta) &= \alpha_{l + 1}^{(a, b)} y^{(a, b)}_{l +
 * 1}(\theta)
 * + \beta_{l + 2}^{(a,b)} y^{(a, b)}_{l + 2}(\theta) + a_l^{(a, b)} \\
 * f_m(\theta, \phi) &= \beta_{l_{\text{min}} + 2}
 * {}_s Y_{l_{\text{min}}, m}(\theta, \phi) y^{(a, b)}_2(\theta) +
 * {}_s Y_{l_{\text{min}} + 1, m}(\theta, \phi) y^{(a, b)}_1(\theta) +
 * a_0^{(a, b)} {}_s Y_{l_{\text{min}}, m}(\theta, \phi),
 * \f}
 *
 * where \f$l_{\text{min}} = \max(|m|, |s|)\f$ is the lowest nonvanishing
 * \f$l\f$ mode for a given \f$m\f$ and \f$s\f$.
 *
 * The coefficients to cache are inferred from a mechanical but lengthy
 * calculation involving the recurrence relation for the Jacobi polynomials.
 * The result is:
 *
 * \f{align*}{
 * \alpha_l^{(a, b)} &= \frac{\sqrt{(2l + 1)(2l - 1)}}
 * {2\sqrt{(l + k)(l + k + a + b)(l + k + a)(l + k + b)}}
 * \left[2 l \cos(\theta) +
 * \frac{a^2 - b^2}{2 l - 2}\right] \\
 * \beta_l^{(a, b)} & = - \sqrt{\frac{(2 l + 1)(l + k + a - 1)(l + k + b - 1)
 * (l + k - 1)(l + k + a + b - 1)}
 * {(2l - 3)(l + k)(l + k + a + b)(l + k + a)(l + k + b)}}
 * \frac{2 l}{2 l - 2},
 * \f}
 * where \f$k = - (a + b)/2\f$ (which is always integral due to the properties
 * of \f$a\f$ and \f$b\f$). Note that because the values of \f$\alpha\f$ and
 * \f$\beta\f$ in the recurrence relation are not needed for any value below
 * \f$l_{\text{min}} + 2\f$, so none of the values in the square-roots or
 * denominators take pathological values for any of the coefficients we require.
 *
 * The \f$\beta\f$ constants are filled in member variable `beta_constant`. The
 * \f$\alpha\f$ values are separately stored as the prefactor for the
 * \f$\cos(\theta)\f$ term and the constant term in `alpha_prefactor` and
 * `alpha_constant` member variables.
 *
 * In addition, it is efficient to cache recurrence coefficients necessary for
 * generating the first couple of spin-weighted spherical harmonic functions for
 * each \f$m\f$ used in the Clenshaw sum.
 *
 * The member variable `harmonic_at_l_min_prefactors` holds the prefactors for
 * directly evaluating the harmonics at \f$s >= m\f$,
 *
 * \f{align*}{
 * {}_s Y_{|s| m}  = {}_s C_{m} e^{i m \phi} \sin^a(\theta/2) \cos^b(\theta/2),
 * \f}
 *
 * where \f${}_s C_m\f$ are the cached prefactors that take the values
 *
 * \f{align*}{
 * {}_s C_m = (-1)^{m + \lambda(m)} \sqrt{\frac{(2 |s| + 1) (|s| + k)!}
 * {4 \pi (|s| + k + a)! (|s| + k + b)!}}
 * \f}
 *
 * and
 *
 * \f{align*}{
 * \lambda(m) =
 * \left\{
 * \begin{array}{ll}
 * 0 &\text{ for } s \ge -m\\
 * s + m &\text{ for } s < -m
 * \end{array} \right..
 * \f}
 *
 * The member variable `harmonic_m_recurrence_prefactors` holds the prefactors
 * necessary to evaluate the lowest harmonics for each \f$m\f$ from the next
 * lowest-in-magnitude \f$m\f$, allowing most leading harmonics to recursively
 * derived.
 *
 * \f{align*}{
 * {}_s Y_{|m|, m} = {}_s D_m  \sin(\theta / 2) \cos(\theta / 2)
 * \left\{
 * \begin{array}{ll}
 * e^{i \phi} {}_s Y_{|m| - 1, m - 1} &\text{ for } m > 0\\
 * e^{-i \phi} {}_s Y_{|m| - 1, m + 1} &\text{ for } m < 0
 * \end{array}\right.,
 * \f}
 *
 * where \f${}_s D_m\f$ are the cached prefactors that take the values
 *
 * \f{align*}{
 * {}_s D_m = -(-1)^{\Delta \lambda(m)} \sqrt{
 * \frac{(2 |m| + 1)(k + |m| + a + b - 1)(k + |m| + a + b)}
 * {(2 |m| - 1)(k + |m| + a)(k + |m| + b)}},
 * \f}
 * and \f$\Delta \lambda(m)\f$ is the difference in \f$\lambda(m)\f$ between
 * the harmonic on the left-hand side and right-hand side of the above
 * recurrence relation, that is \f$\lambda(m) - \lambda(m - 1)\f$ for positive
 * \f$m\f$ and \f$\lambda(m) - \lambda(m + 1)\f$ for negative \f$m\f$.
 *
 * Finally, the member variable
 * `harmonic_at_l_min_plus_one_recurrence_prefactors` holds the prefactors
 * necessary to evaluate parts of the recurrence relations from the lowest
 * \f$l\f$ for a given \f$m\f$ to the next-to-lowest \f$l\f$.
 *
 * \f{align*}{
 *   {}_s Y_{l_{\min} + 1, m} = {}_s E_m
 * \left[(a + 1) + (a + b + 2) \frac{(\cos(\theta) - 1)}{2}\right]
 * {}_s Y_{l_{\min}, m}.
 * \f}
 *
 * where \f${}_s E_m\f$ are the cached prefactors that take the values
 *
 * \f{align*}{
 * {}_s E_{m} = \sqrt{\frac{2 l_{\min} + 3}{ 2 l_{\min} + 1}}
 * \sqrt{\frac{(l_{\min} + k + 1)(l_{\min} + k + a + b + 1)}
 * {(l_{\min} + k + a + 1)(l_{\min} + k + b + 1)}}
 * \f}
 */
class SwshInterpolator {
 public:
  // charm needs the empty constructor
  SwshInterpolator() = default;

  SwshInterpolator(const SwshInterpolator&) = default;
  SwshInterpolator(SwshInterpolator&&) = default;
  SwshInterpolator& operator=(const SwshInterpolator&) = default;
  SwshInterpolator& operator=(SwshInterpolator&&) = default;
  ~SwshInterpolator() = default;

  SwshInterpolator(const DataVector& theta, const DataVector& phi,
                   size_t l_max);

  /*!
   * \brief Perform the Clenshaw recurrence sum, returning by pointer
   * `interpolated` of interpolating the `goldberg_modes` at the collocation
   * points passed to the constructor.
   *
   * \details The core Clenshaw recurrence is in the\f$l\f$ modes of the
   * spin-weighted spherical harmonic. For a set of modes \f$a_l^{(a,b)}\f$ at
   * fixed \f$m\f$, the function value is evaluated by the recurrence:
   *
   * \f{align*}{
   * y^{(a, b)}_{l_\text{max} + 2} &= y^{(a, b)}_{l_\text{max} + 1} = 0 \\
   * y^{(a, b)}_{l}(\theta) &= \alpha_{l + 1}^{(a, b)} y^{(a, b)}_{l +
   * 1}(\theta)
   * + \beta_{l + 2}^{(a,b)} y^{(a, b)}_{l + 2}(\theta) + a_l^{(a, b)} \\
   * f_m(\theta, \phi) &= \beta_{l_{\text{min}} + 2}
   * {}_s Y_{l_{\text{min}}, m}(\theta, \phi) y^{(a, b)}_2(\theta) +
   * {}_s Y_{l_{\text{min}} + 1, m}(\theta, \phi) y^{(a, b)}_1(\theta) +
   * a_0^{(a, b)} {}_s Y_{l_{\text{min}}, m}(\theta, \phi)
   * \f}
   *
   * The recurrence in \f$l\f$ accomplishes much of the work, but for full
   * efficiency, we also recursively evaluate the lowest handful of \f$l\f$s for
   * each \f$m\f$. The details of those additional recurrence tricks can be
   * found in the documentation for `ClenshawRecurrenceConstants`.
   */
  template <int Spin>
  void interpolate(
      gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> interpolated,
      const SpinWeighted<ComplexModalVector, Spin>& goldberg_modes) const;

  /*!
   * \brief Perform the Clenshaw recurrence sum, returning by pointer
   * `interpolated` of interpolating function represented by
   * `libsharp_collocation` at the target points passed to the constructor.
   *
   * \details The core Clenshaw recurrence is in the\f$l\f$ modes of the
   * spin-weighted spherical harmonic. For a set of modes \f$a_l^{(a,b)}\f$ at
   * fixed \f$m\f$, the function value is evaluated by the recurrence:
   *
   * \f{align*}{
   * y^{(a, b)}_{l_\text{max} + 2} &= y^{(a, b)}_{l_\text{max} + 1} = 0 \\
   * y^{(a, b)}_{l}(\theta) &= \alpha_{l + 1}^{(a, b)} y^{(a, b)}_{l +
   * 1}(\theta)
   * + \beta_{l + 2}^{(a,b)} y^{(a, b)}_{l + 2}(\theta) + a_l^{(a, b)} \\
   * f_m(\theta, \phi) &= \beta_{l_{\text{min}} + 2}
   * {}_s Y_{l_{\text{min}}, m}(\theta, \phi) y^{(a, b)}_2(\theta) +
   * {}_s Y_{l_{\text{min}} + 1, m}(\theta, \phi) y^{(a, b)}_1(\theta) +
   * a_0^{(a, b)} {}_s Y_{l_{\text{min}}, m}(\theta, \phi)
   * \f}
   *
   * The recurrence in \f$l\f$ accomplishes much of the work, but for full
   * efficiency, we also recursively evaluate the lowest handful of \f$l\f$s for
   * each \f$m\f$. The details of those additional recurrence tricks can be
   * found in the documentation for `ClenshawRecurrenceConstants`.
   */
  template <int Spin>
  void interpolate(
      gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> interpolated,
      const SpinWeighted<ComplexDataVector, Spin>& libsharp_collocation) const;

  /// \brief Evaluate the SWSH function at the lowest \f$l\f$ value for a given
  /// \f$m\f$ at the target interpolation points.
  ///
  /// \details Included in the public interface for thorough testing, most use
  /// cases should just use the `SwshInterpolator::interpolate()` member
  /// function.
  template <int Spin>
  void direct_evaluation_swsh_at_l_min(
      gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> harmonic,
      int m) const;

  /// \brief Evaluate the SWSH function at the next-to-lowest \f$l\f$ value for
  /// a given \f$m\f$ at the target interpolation points, given input harmonic
  /// values for the lowest \f$l\f$ value.
  ///
  /// \details Included in the public interface for thorough testing, most use
  /// cases should just use the `SwshInterpolator::interpolate()` member
  /// function.
  template <int Spin>
  void evaluate_swsh_at_l_min_plus_one(
      gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> harmonic,
      const SpinWeighted<ComplexDataVector, Spin>& harmonic_at_l_min,
      int m) const;

  /// \brief Evaluate the SWSH function at the lowest \f$l\f$ value for a given
  /// \f$m\f$ at the target interpolation points, given harmonic data at the
  /// next lower \f$m\f$ (by magnitude), passed in by the same pointer used for
  /// the return.
  ///
  /// \details Included in the public interface for thorough testing, most use
  /// cases should just use the `SwshInterpolator::interpolate()` member
  /// function.
  template <int Spin>
  void evaluate_swsh_m_recurrence_at_l_min(
      gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> harmonic,
      int m) const;

  /// \brief Perform the core Clenshaw interpolation at fixed \f$m\f$,
  /// accumulating the result in `interpolation`.
  ///
  /// \details Included in the public interface for thorough testing, most use
  /// cases should just use the `interpolate` member function.
  template <int Spin>
  void clenshaw_sum(
      gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> interpolation,
      const SpinWeighted<ComplexDataVector, Spin>& l_min_harmonic,
      const SpinWeighted<ComplexDataVector, Spin>& l_min_plus_one_harmonic,
      const SpinWeighted<ComplexModalVector, Spin>& goldberg_modes,
      int m) const;

  /// Serialization for Charm++.
  void pup(PUP::er& p);  // NOLINT

 private:
  size_t l_max_ = 0;
  DataVector cos_theta_;
  DataVector sin_theta_;
  DataVector cos_theta_over_two_;
  DataVector sin_theta_over_two_;
  std::vector<DataVector> sin_m_phi_;
  std::vector<DataVector> cos_m_phi_;
  mutable ComplexModalVector raw_libsharp_coefficient_buffer_;
  mutable ComplexModalVector raw_goldberg_coefficient_buffer_;
};
}  // namespace Swsh
}  // namespace Spectral
