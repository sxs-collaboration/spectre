// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function RootFinder::toms748

#pragma once

#include <functional>
#include <iomanip>
#include <ios>
#include <limits>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Simd/Simd.hpp"

namespace RootFinder {
namespace toms748_detail {
// Original implementation of TOMS748 is from Boost:
//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Significant changes made to pretty much all of it to support SIMD by Nils
// Deppe. Changes are copyrighted by SXS Collaboration under the MIT License.

template <typename T>
T safe_div(const T& num, const T& denom, const T& r) {
  if constexpr (std::is_floating_point_v<T>) {
    using std::abs;
    if (abs(denom) < static_cast<T>(1)) {
      if (abs(denom * std::numeric_limits<T>::max()) <= abs(num)) {
        return r;
      }
    }
    return num / denom;
  } else {
    // return num / denom without overflow, return r if overflow would occur.
    const auto mask0 = fabs(denom) < static_cast<T>(1);
    // Note: if denom >= 1.0 you get an FPE because of overflow from
    // `max() * (1. + value)`
    const auto mask = fabs(simd::select(mask0, denom, static_cast<T>(0)) *
                           std::numeric_limits<T>::max()) <= fabs(num);
    return simd::select(mask0 and mask, r,
                        num / simd::select(mask, denom, static_cast<T>(1)));
  }
}

template <typename T>
T secant_interpolate(const T& a, const T& b, const T& fa, const T& fb) {
  //
  // Performs standard secant interpolation of [a,b] given
  // function evaluations f(a) and f(b).  Performs a bisection
  // if secant interpolation would leave us very close to either
  // a or b.  Rationale: we only call this function when at least
  // one other form of interpolation has already failed, so we know
  // that the function is unlikely to be smooth with a root very
  // close to a or b.
  //
  const T tol_batch = std::numeric_limits<T>::epsilon() * static_cast<T>(5);
  // WARNING: There are several different ways to implement the interpolation
  // that all have different rounding properties. Unfortunately this means that
  // tolerances even at 1e-14 can be difficult to achieve generically.
  //
  // `g` below is:
  // const T g = (fa / (fb - fa));
  //
  // const T c = simd::fma((fa / (fb - fa)), (a - b), a); // fails
  //
  // const T c = a * (static_cast<T>(1) - g) + g * b; // works
  //
  // const T c = simd::fma(a, (static_cast<T>(1) - g), g * b); // works
  //
  // Original Boost code:
  // const T c = a - (fa / (fb - fa)) * (b - a); // works

  const T c = a - (fa / (fb - fa)) * (b - a);
  return simd::select((c <= simd::fma(fabs(a), tol_batch, a)) or
                          (c >= simd::fnma(fabs(b), tol_batch, b)),
                      static_cast<T>(0.5) * (a + b), c);
}

template <bool AssumeFinite, typename T>
T quadratic_interpolate(const T& a, const T& b, const T& d, const T& fa,
                        const T& fb, const T& fd,
                        const simd::mask_type_t<T>& incomplete_mask,
                        const unsigned count) {
  // Performs quadratic interpolation to determine the next point,
  // takes count Newton steps to find the location of the
  // quadratic polynomial.
  //
  // Point d must lie outside of the interval [a,b], it is the third
  // best approximation to the root, after a and b.
  //
  // Note: this does not guarantee to find a root
  // inside [a, b], so we fall back to a secant step should
  // the result be out of range.
  //
  // Start by obtaining the coefficients of the quadratic polynomial:
  const T B = safe_div(fb - fa, b - a, std::numeric_limits<T>::max());
  T A = safe_div(fd - fb, d - b, std::numeric_limits<T>::max());
  A = safe_div(A - B, d - a, static_cast<T>(0));

  const auto secant_failure_mask = A == static_cast<T>(0) and incomplete_mask;
  T result_secant{};
  if (UNLIKELY(simd::any(secant_failure_mask))) {
    // failure to determine coefficients, try a secant step:
    result_secant = secant_interpolate(a, b, fa, fb);
    if (UNLIKELY(simd::all(secant_failure_mask or (not incomplete_mask)))) {
      return result_secant;
    }
  }

  // Determine the starting point of the Newton steps:
  //
  // Note: unlike Boost, we assume A*fa doesn't overflow. This speeds up the
  // code quite a bit.
  T c = AssumeFinite
            ? simd::select(A * fa > static_cast<T>(0), a, b)
            : simd::select(simd::sign(A) * simd::sign(fa) > static_cast<T>(0),
                           a, b);

  // Take the Newton steps:
  const T two_A = static_cast<T>(2) * A;
  const T half_a_plus_b = 0.5 * (a + b);
  const T one_minus_a = static_cast<T>(1) - a;
  const T B_minus_A_times_b = B - A * b;
  for (unsigned i = 1; i <= count; ++i) {
    c -= safe_div(simd::fma(simd::fma(A, c, B_minus_A_times_b), c - a, fa),
                  simd::fma(two_A, c - half_a_plus_b, B), one_minus_a + c);
  }
  if (const auto mask = ((c <= a) or (c >= b)) and incomplete_mask;
      simd::any(mask)) {
    // Failure, try a secant step:
    c = simd::select(mask, secant_interpolate(a, b, fa, fb), c);
  }
  return simd::select(secant_failure_mask, result_secant, c);
}

template <bool AssumeFinite, typename T>
T cubic_interpolate(const T& a, const T& b, const T& d, const T& e, const T& fa,
                    const T& fb, const T& fd, const T& fe,
                    const simd::mask_type_t<T>& incomplete_mask) {
  // Uses inverse cubic interpolation of f(x) at points
  // [a,b,d,e] to obtain an approximate root of f(x).
  // Points d and e lie outside the interval [a,b]
  // and are the third and forth best approximations
  // to the root that we have found so far.
  //
  // Note: this does not guarantee to find a root
  // inside [a, b], so we fall back to quadratic
  // interpolation in case of an erroneous result.
  //
  // This commented chunk is the original Boost implementation translated into
  // simd. The actual code below is a heavily optimized version.
  //
  // const T q11 = (d - e) * fd / (fe - fd);
  // const T q21 = (b - d) * fb / (fd - fb);
  // const T q31 = (a - b) * fa / (fb - fa);
  // const T d21 = (b - d) * fd / (fd - fb);
  // const T d31 = (a - b) * fb / (fb - fa);
  //
  // const T q22 = (d21 - q11) * fb / (fe - fb);
  // const T q32 = (d31 - q21) * fa / (fd - fa);
  // const T d32 = (d31 - q21) * fd / (fd - fa);
  // const T q33 = (d32 - q22) * fa / (fe - fa);
  //
  // T c = q31 + q32 + q33 + a;

  // The optimized implementation here is L1-cache bound. That is, we aren't
  // able to completely saturate the FP units because we are waiting on the L1
  // cache. While not ideal, that's okay and just part of the algorithm.
  const T denom_fb_fa = fb - fa;
  const T denom_fd_fb = fd - fb;
  const T denom_fd_fa = fd - fa;
  const T denom_fe_fd = fe - fd;
  const T denom_fe_fb = fe - fb;
  const T denom =
      denom_fe_fb * denom_fe_fd * denom_fd_fa * denom_fd_fb * (fe - fa);

  // Avoid division by zero with mask.
  const T fa_by_denom = fa / simd::select(incomplete_mask, denom_fb_fa * denom,
                                          static_cast<T>(1));

  const T d31 = (a - b);
  const T q21 = (b - d);
  const T q32 = simd::fms(denom_fd_fb, d31, denom_fb_fa * q21) * denom_fe_fb *
                denom_fe_fd;
  const T q22 = simd::fms(denom_fe_fd, q21, (d - e) * denom_fd_fb) * fd *
                denom_fb_fa * denom_fd_fa;

  // Note: the reduction in rounding error that comes from the improvement by
  // Stoer & Bulirsch to Neville's algorithm is adding `a` at the very end as
  // we do below. Alternative ways of evaluating polynomials do not delay this
  // inclusion of `a`, and so then when the correction to `a` is small,
  // floating point errors decrease the accuracy of the result.
  T c = simd::fma(
      fa_by_denom,
      simd::fma(fb, simd::fms(q32, (fe + denom_fd_fa), q22), d31 * denom), a);

  if (const auto mask = ((c <= a) or (c >= b)) and incomplete_mask;
      simd::any(mask)) {
    // Out of bounds step, fall back to quadratic interpolation:
    //
    // Note: we only apply quadratic interpolation at points where cubic
    // failed and that aren't already at a root.
    c = simd::select(
        mask, quadratic_interpolate<AssumeFinite>(a, b, d, fa, fb, fd, mask, 3),
        c);
  }

  return c;
}

template <bool AssumeFinite, typename F, typename T>
void bracket(F f, T& a, T& b, T c, T& fa, T& fb, T& d, T& fd,
             const simd::mask_type_t<T>& incomplete_mask) {
  // Given a point c inside the existing enclosing interval
  // [a, b] sets a = c if f(c) == 0, otherwise finds the new
  // enclosing interval: either [a, c] or [c, b] and sets
  // d and fd to the point that has just been removed from
  // the interval.  In other words d is the third best guess
  // to the root.
  //
  // Note: `bracket` will only modify slots marked as `true` in
  //       `incomplete_mask`
  const T tol_batch = std::numeric_limits<T>::epsilon() * static_cast<T>(2);

  // If the interval [a,b] is very small, or if c is too close
  // to one end of the interval then we need to adjust the
  // location of c accordingly. This is:
  //
  //   if ((b - a) < 2 * tol * a) {
  //     c = a + (b - a) / 2;
  //   } else if (c <= a + fabs(a) * tol) {
  //     c = a + fabs(a) * tol;
  //   } else if (c >= b - fabs(b) * tol) {
  //     c = b - fabs(b) * tol;
  //   }
  const T a_filt = simd::fma(fabs(a), tol_batch, a);
  const T b_filt = simd::fnma(fabs(b), tol_batch, b);
  const T b_minus_a = b - a;
  c = simd::select(
      (static_cast<T>(2) * tol_batch * a > b_minus_a) and incomplete_mask,
      simd::fma(b_minus_a, static_cast<T>(0.5), a),
      simd::select(c <= a_filt, a_filt, simd::select(c >= b_filt, b_filt, c)));

  // Invoke f(c):
  T fc = f(c);

  // if we have a zero then we have an exact solution to the root:
  const auto fc_is_zero_mask = (fc == static_cast<T>(0));
  if (const auto mask = fc_is_zero_mask and incomplete_mask;
      UNLIKELY(simd::any(mask))) {
    a = simd::select(mask, c, a);
    fa = simd::select(mask, static_cast<T>(0), fa);
    d = simd::select(mask, static_cast<T>(0), d);
    fd = simd::select(mask, static_cast<T>(0), fd);
    if (UNLIKELY(simd::all(mask or not incomplete_mask))) {
      return;
    }
  }

  // Non-zero fc, update the interval:
  //
  // Note: unlike Boost, we assume fa*fc doesn't overflow. This speeds up the
  // code quite a bit.
  //
  // Boost code is:
  // if (boost::math::sign(fa) * boost::math::sign(fc) < 0) {...} else {...}
  using simd::sign;
  const auto sign_mask = AssumeFinite
                             ? (fa * fc < static_cast<T>(0))
                             : (sign(fa) * sign(fc) < static_cast<T>(0));
  const auto mask_if =
      (sign_mask and (not fc_is_zero_mask)) and incomplete_mask;
  d = simd::select(mask_if, b, d);
  fd = simd::select(mask_if, fb, fd);
  b = simd::select(mask_if, c, b);
  fb = simd::select(mask_if, fc, fb);

  const auto mask_else =
      ((not sign_mask) and (not fc_is_zero_mask)) and incomplete_mask;
  d = simd::select(mask_else, a, d);
  fd = simd::select(mask_else, fa, fd);
  a = simd::select(mask_else, c, a);
  fa = simd::select(mask_else, fc, fa);
}

template <bool AssumeFinite, class F, class T, class Tol>
std::pair<T, T> toms748_solve(F f, const T& ax, const T& bx, const T& fax,
                              const T& fbx, Tol tol,
                              const simd::mask_type_t<T>& ignore_filter,
                              size_t& max_iter) {
  // Main entry point and logic for Toms Algorithm 748
  // root finder.
  if (UNLIKELY(simd::any(ax >= bx))) {
    throw std::domain_error("Lower bound is larger than upper bound");
  }

  // Sanity check - are we allowed to iterate at all?
  if (UNLIKELY(max_iter == 0)) {
    return std::pair{ax, bx};
  }

  size_t count = max_iter;
  // mu is a parameter in the algorithm that must be between (0, 1).
  static const T mu = 0.5f;

  // initialise a, b and fa, fb:
  T a = ax;
  T b = bx;
  T fa = fax;
  T fb = fbx;

  const auto fa_is_zero_mask = (fa == static_cast<T>(0));
  const auto fb_is_zero_mask = (fb == static_cast<T>(0));
  auto completion_mask =
      tol(a, b) or fa_is_zero_mask or fb_is_zero_mask or ignore_filter;
  auto incomplete_mask = not completion_mask;
  if (UNLIKELY(simd::all(completion_mask))) {
    max_iter = 0;
    return std::pair{simd::select(fb_is_zero_mask, b, a),
                     simd::select(fa_is_zero_mask, a, b)};
  }

  // Note: unlike Boost, we can assume fa*fb doesn't overflow when possible.
  // This speeds up the code quite a bit.
  if (UNLIKELY(simd::any((AssumeFinite ? (fa * fb > static_cast<T>(0))
                                       : (simd::sign(fa) * simd::sign(fb) >
                                          static_cast<T>(0))) and
                         (not fa_is_zero_mask) and (not fb_is_zero_mask)))) {
    throw std::domain_error(
        "Parameters lower and upper bounds do not bracket a root");
  }
  // dummy value for fd, e and fe:
  T fe(static_cast<T>(1e5F));
  T e(static_cast<T>(1e5F));
  T fd(static_cast<T>(1e5F));

  T c(std::numeric_limits<T>::signaling_NaN());
  T d(std::numeric_limits<T>::signaling_NaN());

  const T nan(std::numeric_limits<T>::signaling_NaN());
  auto completed_a = simd::select(completion_mask, a, nan);
  auto completed_b = simd::select(completion_mask, b, nan);
  auto completed_fa = simd::select(completion_mask, fa, nan);
  auto completed_fb = simd::select(completion_mask, fb, nan);
  const auto update_completed = [&fa, &fb, &completion_mask, &incomplete_mask,
                                 &completed_a, &completed_b, &a, &b,
                                 &completed_fa, &completed_fb, &tol]() {
    const auto new_completed =
        (fa == static_cast<T>(0) or tol(a, b)) and (not completion_mask);
    completed_a = simd::select(new_completed, a, completed_a);
    completed_b = simd::select(new_completed, b, completed_b);
    completed_fa = simd::select(new_completed, fa, completed_fa);
    completed_fb = simd::select(new_completed, fb, completed_fb);
    completion_mask = new_completed or completion_mask;
    incomplete_mask = not completion_mask;
    // returns true if _all_ simd registers have been completed
    return simd::all(completion_mask);
  };

  if (simd::any(fa != static_cast<T>(0))) {
    // On the first step we take a secant step:
    c = toms748_detail::secant_interpolate(a, b, fa, fb);
    toms748_detail::bracket<AssumeFinite>(f, a, b, c, fa, fb, d, fd,
                                          incomplete_mask);
    --count;

    // Note: The Boost fa!=0 check is handled with the completion_mask.
    if (count and not update_completed()) {
      // On the second step we take a quadratic interpolation:
      c = toms748_detail::quadratic_interpolate<AssumeFinite>(
          a, b, d, fa, fb, fd, incomplete_mask, 2);
      e = d;
      fe = fd;
      toms748_detail::bracket<AssumeFinite>(f, a, b, c, fa, fb, d, fd,
                                            incomplete_mask);
      --count;
      update_completed();
    }
  }

  T u(std::numeric_limits<T>::signaling_NaN());
  T fu(std::numeric_limits<T>::signaling_NaN());
  T a0(std::numeric_limits<T>::signaling_NaN());
  T b0(std::numeric_limits<T>::signaling_NaN());

  // Note: The Boost fa!=0 check is handled with the completion_mask.
  while (count and not simd::all(completion_mask)) {
    // save our brackets:
    a0 = a;
    b0 = b;
    // Starting with the third step taken
    // we can use either quadratic or cubic interpolation.
    // Cubic interpolation requires that all four function values
    // fa, fb, fd, and fe are distinct, should that not be the case
    // then variable prof will get set to true, and we'll end up
    // taking a quadratic step instead.
    static const T min_diff = std::numeric_limits<T>::min() * 32;
    bool prof =
        simd::any(((fabs(fa - fb) < min_diff) or (fabs(fa - fd) < min_diff) or
                   (fabs(fa - fe) < min_diff) or (fabs(fb - fd) < min_diff) or
                   (fabs(fb - fe) < min_diff) or (fabs(fd - fe) < min_diff)) and
                  incomplete_mask);
    if (prof) {
      c = toms748_detail::quadratic_interpolate<AssumeFinite>(
          a, b, d, fa, fb, fd, incomplete_mask, 2);
    } else {
      c = toms748_detail::cubic_interpolate<AssumeFinite>(
          a, b, d, e, fa, fb, fd, fe, incomplete_mask);
    }
    // re-bracket, and check for termination:
    e = d;
    fe = fd;
    toms748_detail::bracket<AssumeFinite>(f, a, b, c, fa, fb, d, fd,
                                          incomplete_mask);
    if ((0 == --count) or update_completed()) {
      break;
    }
    // Now another interpolated step:
    prof =
        simd::any(((fabs(fa - fb) < min_diff) or (fabs(fa - fd) < min_diff) or
                   (fabs(fa - fe) < min_diff) or (fabs(fb - fd) < min_diff) or
                   (fabs(fb - fe) < min_diff) or (fabs(fd - fe) < min_diff)) and
                  incomplete_mask);
    if (prof) {
      c = toms748_detail::quadratic_interpolate<AssumeFinite>(
          a, b, d, fa, fb, fd, incomplete_mask, 3);
    } else {
      c = toms748_detail::cubic_interpolate<AssumeFinite>(
          a, b, d, e, fa, fb, fd, fe, incomplete_mask);
    }
    // Bracket again, and check termination condition, update e:
    toms748_detail::bracket<AssumeFinite>(f, a, b, c, fa, fb, d, fd,
                                          incomplete_mask);
    if ((0 == --count) or update_completed()) {
      break;
    }

    // Now we take a double-length secant step:
    const auto fabs_fa_less_fabs_fb_mask =
        (fabs(fa) < fabs(fb)) and incomplete_mask;
    u = simd::select(fabs_fa_less_fabs_fb_mask, a, b);
    fu = simd::select(fabs_fa_less_fabs_fb_mask, fa, fb);
    const T b_minus_a = b - a;
    // Assumes that bounds a & b are not so close that fa == fb. Boost makes
    // this assumption too. If this is violated then the algorithm doesn't
    // work since the function at least appears constant.
    c = simd::fnma(static_cast<T>(2) * (fu / (fb - fa)), b_minus_a, u);
    c = simd::select(static_cast<T>(2) * fabs(c - u) > b_minus_a,
                     simd::fma(static_cast<T>(0.5), b_minus_a, a), c);

    // Bracket again, and check termination condition:
    e = d;
    fe = fd;
    toms748_detail::bracket<AssumeFinite>(f, a, b, c, fa, fb, d, fd,
                                          incomplete_mask);
    if ((0 == --count) or update_completed()) {
      break;
    }

    // And finally... check to see if an additional bisection step is
    // to be taken, we do this if we're not converging fast enough:
    const auto bisection_mask = (b - a) >= mu * (b0 - a0) and incomplete_mask;
    if (LIKELY(simd::none(bisection_mask))) {
      continue;
    }
    // bracket again on a bisection:
    //
    // Note: the mask ensures we only ever modify the slots that the mask has
    // identified as needing to be modified.
    e = simd::select(bisection_mask, d, e);
    fe = simd::select(bisection_mask, fd, fe);
    toms748_detail::bracket<AssumeFinite>(
        f, a, b, simd::fma((b - a), static_cast<T>(0.5), a), fa, fb, d, fd,
        bisection_mask);
    --count;
    if (update_completed()) {
      break;
    }
  }  // while loop

  max_iter -= count;
  completed_b =
      simd::select(completed_fa == static_cast<T>(0), completed_a, completed_b);
  completed_a =
      simd::select(completed_fb == static_cast<T>(0), completed_b, completed_a);
  return std::pair{completed_a, completed_b};
}
}  // namespace toms748_detail

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method.
 *
 * `f` is a unary invokable that takes a `double` which is the current value at
 * which to evaluate `f`. An example is below.
 *
 * \snippet Test_TOMS748.cpp double_root_find
 *
 * The TOMS_748 algorithm searches for a root in the interval [`lower_bound`,
 * `upper_bound`], and will throw if this interval does not bracket a root,
 * i.e. if `f(lower_bound) * f(upper_bound) > 0`.
 *
 * The arguments `f_at_lower_bound` and `f_at_upper_bound` are optional, and
 * are the function values at `lower_bound` and `upper_bound`. These function
 * values are often known because the user typically checks if a root is
 * bracketed before calling `toms748`; passing the function values here saves
 * two function evaluations.
 *
 * \note if `AssumeFinite` is true than the code assumes all numbers are
 * finite and that `a > 0` is equivalent to `sign(a) > 0`. This reduces
 * runtime but will cause bugs if the numbers aren't finite. It also assumes
 * that products like `fa * fb` are also finite.
 *
 * \requires Function `f` is invokable with a `double`
 *
 * \throws `convergence_error` if the requested tolerance is not met after
 *                            `max_iterations` iterations.
 */
template <bool AssumeFinite = false, typename Function, typename T>
T toms748(const Function& f, const T lower_bound, const T upper_bound,
          const T f_at_lower_bound, const T f_at_upper_bound,
          const simd::scalar_type_t<T> absolute_tolerance,
          const simd::scalar_type_t<T> relative_tolerance,
          const size_t max_iterations = 100,
          const simd::mask_type_t<T> ignore_filter =
              static_cast<simd::mask_type_t<T>>(0)) {
  ASSERT(relative_tolerance >
             std::numeric_limits<simd::scalar_type_t<T>>::epsilon(),
         "The relative tolerance is too small. Got "
             << relative_tolerance << " but must be at least "
             << std::numeric_limits<simd::scalar_type_t<T>>::epsilon());
  if (simd::any(f_at_lower_bound * f_at_upper_bound > 0.0)) {
    ERROR("Root not bracketed: f(" << lower_bound << ") = " << f_at_lower_bound
                                   << ", f(" << upper_bound
                                   << ") = " << f_at_upper_bound);
  }

  std::size_t max_iters = max_iterations;

  // This solver requires tol to be passed as a termination condition. This
  // termination condition is equivalent to the convergence criteria used by the
  // GSL
  const auto tol = [absolute_tolerance, relative_tolerance](const T& lhs,
                                                            const T& rhs) {
    return simd::abs(lhs - rhs) <=
           simd::fma(T(relative_tolerance),
                     simd::min(simd::abs(lhs), simd::abs(rhs)),
                     T(absolute_tolerance));
  };
  auto result = toms748_detail::toms748_solve<AssumeFinite>(
      f, lower_bound, upper_bound, f_at_lower_bound, f_at_upper_bound, tol,
      ignore_filter, max_iters);
  if (max_iters >= max_iterations) {
    throw convergence_error(
        MakeString{}
        << std::setprecision(8) << std::scientific
        << "toms748 reached max iterations without converging.\nAbsolute "
           "tolerance: "
        << absolute_tolerance << "\nRelative tolerance: " << relative_tolerance
        << "\nResult: " << get_output(result.first) << " "
        << get_output(result.second));
  }
  return simd::fma(static_cast<T>(0.5), (result.second - result.first),
                   result.first);
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method, where
 * function values are not supplied at the lower and upper bounds.
 *
 * \note if `AssumeFinite` is true than the code assumes all numbers are
 * finite and that `a > 0` is equivalent to `sign(a) > 0`. This reduces
 * runtime but will cause bugs if the numbers aren't finite. It also assumes
 * that products like `fa * fb` are also finite.
 */
template <bool AssumeFinite = false, typename Function, typename T>
T toms748(const Function& f, const T lower_bound, const T upper_bound,
          const simd::scalar_type_t<T> absolute_tolerance,
          const simd::scalar_type_t<T> relative_tolerance,
          const size_t max_iterations = 100,
          const simd::mask_type_t<T> ignore_filter =
              static_cast<simd::mask_type_t<T>>(0)) {
  return toms748<AssumeFinite>(
      f, lower_bound, upper_bound, f(lower_bound), f(upper_bound),
      absolute_tolerance, relative_tolerance, max_iterations, ignore_filter);
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method on each
 * element in a `DataVector`.
 *
 * `f` is a binary invokable that takes a `double` as its first argument and a
 * `size_t` as its second. The `double` is the current value at which to
 * evaluate `f`, and the `size_t` is the current index into the `DataVector`s.
 * Below is an example of how to root find different functions by indexing into
 * a lambda-captured `DataVector` using the `size_t` passed to `f`.
 *
 * \snippet Test_TOMS748.cpp datavector_root_find
 *
 * For each index `i` into the DataVector, the TOMS_748 algorithm searches for a
 * root in the interval [`lower_bound[i]`, `upper_bound[i]`], and will throw if
 * this interval does not bracket a root,
 * i.e. if `f(lower_bound[i], i) * f(upper_bound[i], i) > 0`.
 *
 * See the [Boost](http://www.boost.org/) documentation for more details.
 *
 * \requires Function `f` be callable with a `double` and a `size_t`
 *
 * \note if `AssumeFinite` is true than the code assumes all numbers are
 * finite and that `a > 0` is equivalent to `sign(a) > 0`. This reduces
 * runtime but will cause bugs if the numbers aren't finite. It also assumes
 * that products like `fa * fb` are also finite.
 *
 * \throws `convergence_error` if, for any index, the requested tolerance is not
 * met after `max_iterations` iterations.
 */
template <bool UseSimd = true, bool AssumeFinite = false, typename Function>
DataVector toms748(const Function& f, const DataVector& lower_bound,
                   const DataVector& upper_bound,
                   const double absolute_tolerance,
                   const double relative_tolerance,
                   const size_t max_iterations = 100) {
  DataVector result_vector{lower_bound.size()};
  if constexpr (UseSimd) {
    constexpr size_t simd_width{simd::size<
        std::decay_t<decltype(simd::load_unaligned(lower_bound.data()))>>()};
    const size_t vectorized_size =
        lower_bound.size() - lower_bound.size() % simd_width;
    for (size_t i = 0; i < vectorized_size; i += simd_width) {
      simd::store_unaligned(
          &result_vector[i],
          toms748<AssumeFinite>([&f, i](const auto x) { return f(x, i); },
                                simd::load_unaligned(&lower_bound[i]),
                                simd::load_unaligned(&upper_bound[i]),
                                absolute_tolerance, relative_tolerance,
                                max_iterations));
    }
    for (size_t i = vectorized_size; i < lower_bound.size(); ++i) {
      result_vector[i] = toms748<AssumeFinite>(
          [&f, i](const auto x) { return f(x, i); }, lower_bound[i],
          upper_bound[i], absolute_tolerance, relative_tolerance,
          max_iterations);
    }
  } else {
    for (size_t i = 0; i < result_vector.size(); ++i) {
      result_vector[i] = toms748<AssumeFinite>(
          [&f, i](double x) { return f(x, i); }, lower_bound[i], upper_bound[i],
          absolute_tolerance, relative_tolerance, max_iterations);
    }
  }
  return result_vector;
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method on each
 * element in a `DataVector`, where function values are supplied at the lower
 * and upper bounds.
 *
 * Supplying function values is an optimization that saves two
 * function calls per point.  The function values are often available
 * because one often checks if the root is bracketed before calling `toms748`.
 *
 * \note if `AssumeFinite` is true than the code assumes all numbers are
 * finite and that `a > 0` is equivalent to `sign(a) > 0`. This reduces
 * runtime but will cause bugs if the numbers aren't finite. It also assumes
 * that products like `fa * fb` are also finite.
 */
template <bool UseSimd = true, bool AssumeFinite = false, typename Function>
DataVector toms748(const Function& f, const DataVector& lower_bound,
                   const DataVector& upper_bound,
                   const DataVector& f_at_lower_bound,
                   const DataVector& f_at_upper_bound,
                   const double absolute_tolerance,
                   const double relative_tolerance,
                   const size_t max_iterations = 100) {
  DataVector result_vector{lower_bound.size()};
  if constexpr (UseSimd) {
    constexpr size_t simd_width{simd::size<
        std::decay_t<decltype(simd::load_unaligned(lower_bound.data()))>>()};
    const size_t vectorized_size =
        lower_bound.size() - lower_bound.size() % simd_width;
    for (size_t i = 0; i < vectorized_size; i += simd_width) {
      simd::store_unaligned(
          &result_vector[i],
          toms748<AssumeFinite>([&f, i](const auto x) { return f(x, i); },
                                simd::load_unaligned(&lower_bound[i]),
                                simd::load_unaligned(&upper_bound[i]),
                                simd::load_unaligned(&f_at_lower_bound[i]),
                                simd::load_unaligned(&f_at_upper_bound[i]),
                                absolute_tolerance, relative_tolerance,
                                max_iterations));
    }
    for (size_t i = vectorized_size; i < lower_bound.size(); ++i) {
      result_vector[i] = toms748<AssumeFinite>(
          [&f, i](double x) { return f(x, i); }, lower_bound[i], upper_bound[i],
          f_at_lower_bound[i], f_at_upper_bound[i], absolute_tolerance,
          relative_tolerance, max_iterations);
    }
  } else {
    for (size_t i = 0; i < lower_bound.size(); ++i) {
      result_vector[i] = toms748<AssumeFinite>(
          [&f, i](double x) { return f(x, i); }, lower_bound[i], upper_bound[i],
          f_at_lower_bound[i], f_at_upper_bound[i], absolute_tolerance,
          relative_tolerance, max_iterations);
    }
  }
  return result_vector;
}
}  // namespace RootFinder
