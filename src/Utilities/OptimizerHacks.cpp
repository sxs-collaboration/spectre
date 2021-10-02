// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/OptimizerHacks.hpp"

#if defined(__clang__) && __clang__ < 11
namespace optimizer_hacks {
void indicate_value_can_be_changed(void* /*variable*/) {}
}  // namespace optimizer_hacks
#endif  /* defined(__clang__) && __clang__ < 11 */
