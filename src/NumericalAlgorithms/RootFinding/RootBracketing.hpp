// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace RootFinder {
namespace bracketing_detail {
// Brackets a root, given a functor f(x) that returns a
// std::optional<double> and given two arrays x and y (with y=f(x))
// containing points that have already been tried for bracketing.
//
// Returns a std::array<double,4> containing {{x1,x2,y1,y2}} where
// x1 and x2 bracket the root, and y1=f(x1) and y2=f(x2).
//
// Note that y might be undefined (i.e. an invalid std::optional)
// for values of x at or near the endpoints of the interval.  We
// assume that if f(x1) and f(x2) are valid for some x1 and x2, then
// f(x) is valid for all x between x1 and x2.
//
// Assumes that there is a root between the first and last points of
// the array.
// We also assume that there is only one root.
//
// So this means we have only 2 possibilities for the validity of the
// points in the input x,y arrays:
// 1) All points are invalid, e.g. "X X X X X X X X".
//    (here X represents an invalid point)
// 2) All valid points are adjacent, with the same sign, e.g. "X X o o X X"
//    or "o o X X X" or "X X X o o".
//    (here o represents a valid point)
// Note that we assume that all valid points have the same sign; otherwise
// the caller would have known that the root was bracketed and the caller would
// not have called bracket_by_contracting.
//
// Also note that we exclude the case "o o o o o" (no roots), since the
// caller would have known that too.  If such a case is found, an error is
// thrown.  An error is also thrown if the size of the region where the
// sign changes is so small that the number of iterations is exceeded.
//
// For case 1) above, we bisect each pair of points, and call
// bracket_by_contracting recursively until we find a valid point.
// For case 2) above, it is sufficent to check for a bracket only
// between valid and invalid points.  That is, for "X X + + X X" we
// check only between points 1 and 2 and between points 3 and 4 (where
// points are numbered starting from zero).  For "+ + X X X" we check
// only between points 1 and 2.
template <typename Functor>
std::array<double, 4> bracket_by_contracting(
    const std::vector<double>& x, const std::vector<std::optional<double>>& y,
    const Functor& f, const size_t level = 0) {
  constexpr size_t max_level = 6;
  if (level > max_level) {
    ERROR("Too many iterations in bracket_by_contracting. Either refine the "
          "initial range/guess or increase max_level.");
  }

  // First check if we have any valid points.
  size_t last_valid_index = y.size();
  for (size_t i = y.size(); i >= 1; --i) {
    if (y[i - 1].has_value()) {
      last_valid_index = i - 1;
      break;
    }
  }

  if (last_valid_index == y.size()) {
    // No valid points!

    // Create larger arrays with one point between each of the already
    // computed points.
    std::vector<double> bisected_x(x.size() * 2 - 1);
    std::vector<std::optional<double>> bisected_y(y.size() * 2 - 1);

    // Copy all even-numbered points in the range.
    for (size_t i = 0; i < x.size(); ++i) {
      bisected_x[2 * i] = x[i];
      bisected_y[2 * i] = y[i];
    }

    // Fill midpoints and check for bracket on each one.
    for (size_t i = 0; i < x.size() - 1; ++i) {
      bisected_x[2 * i + 1] = x[i] + 0.5 * (x[i + 1] - x[i]);
      bisected_y[2 * i + 1] = f(bisected_x[2 * i + 1]);
      if (bisected_y[2 * i + 1].has_value()) {
        // Valid point! We know that all the other points are
        // invalid, so we need to check only 3 points in the next
        // iteration: the new valid point and its neighbors.
        return bracket_by_contracting({{x[i], bisected_x[2 * i + 1], x[i + 1]}},
                                      {{y[i], bisected_y[2 * i + 1], y[i + 1]}},
                                      f, level + 1);
      }
    }
    // We still have no valid points. So recurse, using all points.
    // The next iteration will bisect all the points.
    return bracket_by_contracting(bisected_x, bisected_y, f, level + 1);
  }

  // If we get here, we have found a valid point; in particular we have
  // found the last valid point in the array.

  // Find the first valid point in the array.
  size_t first_valid_index = 0;
  for (size_t i = 0; i < y.size(); ++i) {
    if (y[i].has_value()) {
      first_valid_index = i;
      break;
    }
  }

  // Make a new set of points that includes only the points that
  // neighbor the boundary between valid and invalid points.
  std::vector<double> x_near_valid_point;
  std::vector<std::optional<double>> y_near_valid_point;

  if (first_valid_index == 0 and last_valid_index == y.size() - 1) {
    ERROR(
        "bracket_while_contracting: found a case where all points are valid,"
        "which should not happen under our assumptions.");
  }

  if (first_valid_index > 0) {
    // Check for a root between first_valid_index-1 and first_valid_index.
    const double x_test =
        x[first_valid_index - 1] +
        0.5 * (x[first_valid_index] - x[first_valid_index - 1]);
    const auto y_test = f(x_test);
    if (y_test.has_value() and
        y[first_valid_index].value() * y_test.value() <= 0.0) {
      // Bracketed!
      return std::array<double, 4>{{x_test, x[first_valid_index],
                                    y_test.value(),
                                    y[first_valid_index].value()}};
    } else {
      x_near_valid_point.push_back(x[first_valid_index - 1]);
      y_near_valid_point.push_back(y[first_valid_index - 1]);
      x_near_valid_point.push_back(x_test);
      y_near_valid_point.push_back(y_test);
      x_near_valid_point.push_back(x[first_valid_index]);
      y_near_valid_point.push_back(y[first_valid_index]);
    }
  }
  if (last_valid_index < y.size() - 1) {
    // Check for a root between last_valid_index and last_valid_index+1.
    const double x_test = x[last_valid_index] +
                          0.5 * (x[last_valid_index + 1] - x[last_valid_index]);
    const auto y_test = f(x_test);
    if (y_test.has_value() and
        y[last_valid_index].value() * y_test.value() <= 0.0) {
      // Bracketed!
      return std::array<double, 4>{{x[last_valid_index], x_test,
                                    y[last_valid_index].value(),
                                    y_test.value()}};
    } else {
      if (first_valid_index != last_valid_index or first_valid_index == 0) {
        x_near_valid_point.push_back(x[last_valid_index]);
        y_near_valid_point.push_back(y[last_valid_index]);
      }  // else we already pushed back last_valid_index (==first_valid_index).
      x_near_valid_point.push_back(x_test);
      y_near_valid_point.push_back(y_test);
      x_near_valid_point.push_back(x[last_valid_index + 1]);
      y_near_valid_point.push_back(y[last_valid_index + 1]);
    }
  }

  // We have one or more valid points but we didn't find a bracket.
  // That is, we have something like "X X o o X X" or "X X o o" or "o o X X".
  // So recurse, zooming in to the boundary (either one boundary or two
  // boundaries) between valid and invalid points.
  // Note that "o o o o" is prohibited by our assumptions, and checked for
  // above just in case it occurs by mistake.
  return bracket_by_contracting(x_near_valid_point, y_near_valid_point, f,
                                level + 1);
}
}  // namespace bracketing_detail

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Brackets the root of the function `f`, assuming a single
 * root in a given interval \f$f[x_\mathrm{lo},x_\mathrm{up}]\f$
 * and assuming that `f` is defined only in an unknown smaller
 * interval \f$f[x_a,x_b]\f$ where
 * \f$x_\mathrm{lo} \leq x_a \leq x_b \leq x_\mathrm{hi}\f$.
 *
 * `f` is a unary invokable that takes a `double` which is the current value at
 * which to evaluate `f`.  `f` returns a `std::optional<double>` which
 * evaluates to false if the function is undefined at the supplied point.
 *
 * Assumes that there is only one root in the interval.
 *
 * Assumes that if \f$f(x_1)\f$ and \f$f(x_2)\f$ are both defined for
 * some \f$(x_1,x_2)\f$, then \f$f(x)\f$ is defined for all \f$x\f$
 * between \f$x_1\f$ and \f$x_2\f$.
 *
 * On input, assumes that the root lies in the interval
 * [`lower_bound`,`upper_bound`].  Optionally takes a `guess` for the
 * location of the root.  If `guess` is supplied, then evaluates the
 * function first at `guess` and `upper_bound` before trying
 * `lower_bound`: this means that it would be optimal if `guess`
 * underestimates the actual root and if `upper_bound` was less likely
 * to be undefined than `lower_bound`.
 *
 * On return, `lower_bound` and `upper_bound` are replaced with values that
 * bracket the root and for which the function is defined, and
 * `f_at_lower_bound` and `f_at_upper_bound` are replaced with
 * `f` evaluated at those bracketing points.
 *
 * `bracket_possibly_undefined_function_in_interval` throws an error if
 *  all points are valid but of the same sign (because that would indicate
 *  multiple roots but we assume only one root), if no root exists, or
 *  if the range of a sign change is sufficently small relative to the
 *  given interval that the number of iterations to find the root is exceeded.
 *
 */
template <typename Functor>
void bracket_possibly_undefined_function_in_interval(
    const gsl::not_null<double*> lower_bound,
    const gsl::not_null<double*> upper_bound,
    const gsl::not_null<double*> f_at_lower_bound,
    const gsl::not_null<double*> f_at_upper_bound, const Functor& f,
    const double guess) {
  // Initial values of x1,x2,y1,y2.  Use `guess` and `upper_bound`,
  // because in typical usage `guess` underestimates the actual
  // root, and `lower_bound` is more likely than `upper_bound` to be
  // invalid.
  double x1 = guess;
  double x2 = *upper_bound;
  auto y1 = f(x1);
  auto y2 = f(x2);
  const bool y1_defined = y1.has_value();
  const bool y2_defined = y2.has_value();
  if (not(y1_defined and y2_defined and y1.value() * y2.value() <= 0.0)) {
    // Root is not bracketed.
    // Before moving to the general algorithm, try the remaining
    // input point that was supplied.
    const double x3 = *lower_bound;
    const auto y3 = f(x3);
    const bool y3_defined = y3.has_value();
    if (y1_defined and y3_defined and y1.value() * y3.value() <= 0.0) {
      // Bracketed! Throw out x2,y2.  Rename variables to keep x1 < x2.
      x2 = x1;
      y2 = y1;
      x1 = x3;
      y1 = y3;
    } else {
      // Our simple checks didn't work, so call the more general method.
      // There are 8 cases:
      //
      // y3 y1 y2
      // --------
      // X  X  X
      // o  X  X
      // X  o  X
      // o  o  X
      // X  o  o
      // o  o  o
      // X  X  o
      // o  X  o
      //
      // where X means an invalid point, o means a valid point.
      // All valid points have the same sign, or we would have found a
      // bracket already.
      //
      // Before calling the general case, error on "o o o" and "o X o".
      // Both of these are prohibited by our assumptions (we
      // assume the root is in the interval so no "o o o", and we
      // assume that all invalid points are at the end of interval, so no
      // "o X o").
      if (y2_defined and y3_defined) {
        ERROR(
            "bracket_possibly_undefined_function_in_interval: found "
            "case that should not happen under our assumptions.");
      }
      std::array<double, 4> tmp = bracketing_detail::bracket_by_contracting(
          {{x3, x1, x2}}, {{y3, y1, y2}}, f);
      x1 = tmp[0];
      x2 = tmp[1];
      y1 = tmp[2];
      y2 = tmp[3];
    }
  }
  *f_at_lower_bound = y1.value();
  *f_at_upper_bound = y2.value();
  *lower_bound = x1;
  *upper_bound = x2;
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Brackets the single root of the
 * function `f` for each element in a `DataVector`, assuming the root
 * lies in the given interval and that `f` may be undefined at some
 * points in the interval.
 *
 * `f` is a binary invokable that takes a `double` and a `size_t` as
 * arguments.  The `double` is the current value at which to evaluate
 * `f`, and the `size_t` is the index into the `DataVector`s.  `f`
 * returns a `std::optional<double>` which evaluates to false if the
 * function is undefined at the supplied point.
 *
 * Assumes that there is only one root in the interval.
 *
 * Assumes that if \f$f(x_1)\f$ and \f$f(x_2)\f$ are both defined for
 * some \f$(x_1,x_2)\f$, then \f$f(x)\f$ is defined for all \f$x\f$
 * between \f$x_1\f$ and \f$x_2\f$.
 *
 * On input, assumes that the root lies in the interval
 * [`lower_bound`,`upper_bound`].  Optionally takes a `guess` for the
 * location of the root.
 *
 * On return, `lower_bound` and `upper_bound` are replaced with values that
 * bracket the root and for which the function is defined, and
 * `f_at_lower_bound` and `f_at_upper_bound` are replaced with
 * `f` evaluated at those bracketing points.
 *
 */
template <typename Functor>
void bracket_possibly_undefined_function_in_interval(
    const gsl::not_null<DataVector*> lower_bound,
    const gsl::not_null<DataVector*> upper_bound,
    const gsl::not_null<DataVector*> f_at_lower_bound,
    const gsl::not_null<DataVector*> f_at_upper_bound, const Functor& f,
    const DataVector& guess) {
  for (size_t s = 0; s < lower_bound->size(); ++s) {
    bracket_possibly_undefined_function_in_interval(
        &((*lower_bound)[s]), &((*upper_bound)[s]), &((*f_at_lower_bound)[s]),
        &((*f_at_upper_bound)[s]), [&f, &s](const double x) { return f(x, s); },
        guess[s]);
  }
}

/*
 * Version of `bracket_possibly_undefined_function_in_interval`
 * without a supplied initial guess; uses the mean of `lower_bound` and
 * `upper_bound` as the guess.
 */
template <typename Functor>
void bracket_possibly_undefined_function_in_interval(
    const gsl::not_null<double*> lower_bound,
    const gsl::not_null<double*> upper_bound,
    const gsl::not_null<double*> f_at_lower_bound,
    const gsl::not_null<double*> f_at_upper_bound, const Functor& f) {
  bracket_possibly_undefined_function_in_interval(
      lower_bound, upper_bound, f_at_lower_bound, f_at_upper_bound, f,
      *lower_bound + 0.5 * (*upper_bound - *lower_bound));
}

/*
 * Version of `bracket_possibly_undefined_function_in_interval`
 * without a supplied initial guess; uses the mean of `lower_bound` and
 * `upper_bound` as the guess.
 */
template <typename Functor>
void bracket_possibly_undefined_function_in_interval(
    const gsl::not_null<DataVector*> lower_bound,
    const gsl::not_null<DataVector*> upper_bound,
    const gsl::not_null<DataVector*> f_at_lower_bound,
    const gsl::not_null<DataVector*> f_at_upper_bound, const Functor& f) {
  bracket_possibly_undefined_function_in_interval(
      lower_bound, upper_bound, f_at_lower_bound, f_at_upper_bound, f,
      *lower_bound + 0.5 * (*upper_bound - *lower_bound));
}
}  // namespace RootFinder
