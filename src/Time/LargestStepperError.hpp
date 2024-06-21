// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "DataStructures/SpinWeighted.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsComplexOfFundamental.hpp"
#include "Utilities/TypeTraits/IsIterable.hpp"

/// \cond
template <typename TagsList>
class Variables;
/// \endcond

/*!
 * \ingroup TimeGroup
 * \brief Calculate the pointwise worst error.
 *
 * \details For a `double`, or `std::complex<double>`, calculates
 *
 * \f{equation}
 *   \frac{e}{a + r \max(|v|, |v + e|)}
 * \f}
 *
 * where $v$ is \p values, $e$ is \p errors, and $a$ and $r$ are the
 * tolerances from \p tolerances.
 *
 * For iterable or `Variables` types, takes the largest value over all
 * the points.
 *
 * For `SpinWeighted` types, returns the result for the contained values.
 */
template <typename EvolvedType, typename ErrorType>
double largest_stepper_error(const EvolvedType& values, const ErrorType& errors,
                             const StepperErrorTolerances& tolerances) {
  if constexpr (std::is_fundamental_v<std::remove_cv_t<EvolvedType>> or
                tt::is_complex_of_fundamental_v<
                    std::remove_cv_t<EvolvedType>>) {
    return std::abs(errors) /
           (tolerances.absolute +
            tolerances.relative * std::max(abs(values), abs(values + errors)));
  } else if constexpr (tt::is_iterable_v<std::remove_cv_t<EvolvedType>>) {
    double result = 0.0;
    double recursive_call_result;
    for (auto val_it = values.begin(), err_it = errors.begin();
         val_it != values.end(); ++val_it, ++err_it) {
      recursive_call_result =
          largest_stepper_error(*val_it, *err_it, tolerances);
      if (recursive_call_result > result) {
        result = recursive_call_result;
      }
    }
    return result;
  } else if constexpr (tt::is_a_v<Variables, std::remove_cv_t<EvolvedType>>) {
    double result = 0.0;
    tmpl::for_each<typename EvolvedType::tags_list>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          const double recursive_call_result = largest_stepper_error(
              get<Tag>(values), get<Tag>(errors), tolerances);
          if (recursive_call_result > result) {
            result = recursive_call_result;
          }
        });
    return result;
  } else if constexpr (is_any_spin_weighted_v<std::remove_cv_t<EvolvedType>>) {
    return largest_stepper_error(values.data(), errors.data(), tolerances);
  }
}
