// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/IsNan.hpp"

#include <cmath>

#include "ErrorHandling/FloatingPointExceptions.hpp"

bool is_nan(const double arg) {
  disable_floating_point_exceptions();
#if defined(__clang__) && __clang__ < 10
  // Old versions of clang still throw FPEs here, so prevent then from
  // reordering statements.
  asm("");
#endif  /* defined(__clang__) && __clang__ < 10 */
  const bool result = std::isnan(arg);
#if defined(__clang__) && __clang__ < 10
  asm("");
#endif  /* defined(__clang__) && __clang__ < 10 */
  enable_floating_point_exceptions();
  return result;
}

bool is_nan(const float arg) {
  disable_floating_point_exceptions();
#if defined(__clang__) && __clang__ < 10
  // Old versions of clang still throw FPEs here, so prevent then from
  // reordering statements.
  asm("");
#endif  /* defined(__clang__) && __clang__ < 10 */
  const bool result = std::isnan(arg);
#if defined(__clang__) && __clang__ < 10
  asm("");
#endif  /* defined(__clang__) && __clang__ < 10 */
  enable_floating_point_exceptions();
  return result;
}
