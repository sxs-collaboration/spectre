// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/IsNan.hpp"

#include <cmath>

#include "ErrorHandling/FloatingPointExceptions.hpp"

bool isnan_safe(const double arg) {
  //disable_floating_point_exceptions();
#if defined(__clang__) && __clang__ < 10
  // Old versions of clang still throw FPEs here, so prevent them from
  // reordering statements.
  //asm("");
#endif  /* defined(__clang__) && __clang__ < 10 */
  const bool result = std::isnan(arg);
#if defined(__clang__) && __clang__ < 10
  //asm("");
#endif  /* defined(__clang__) && __clang__ < 10 */
  //enable_floating_point_exceptions();
  return result;
}

bool isnan_safe(const float arg) {
  //disable_floating_point_exceptions();
#if defined(__clang__) && __clang__ < 10
  // Old versions of clang still throw FPEs here, so prevent them from
  // reordering statements.
  //asm("");
#endif  /* defined(__clang__) && __clang__ < 10 */
  const bool result = std::isnan(arg);
#if defined(__clang__) && __clang__ < 10
  //asm("");
#endif  /* defined(__clang__) && __clang__ < 10 */
  //enable_floating_point_exceptions();
  return result;
}
