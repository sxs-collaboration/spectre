// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#if defined(__clang__) && __clang__ < 11
/// \ingroup PeoGroup
/// Workarounds for optimizer bugs
namespace optimizer_hacks {
/// \ingroup PeoGroup
/// Produce a situation where the value of `*variable` could have been
/// changed without the optimizer's knowledge, without actually
/// changing the variable.
void indicate_value_can_be_changed(void* variable);
}  // namespace optimizer_hacks

#define VARIABLE_CAUSES_CLANG_FPE(var) \
  ::optimizer_hacks::indicate_value_can_be_changed(&(var))
#else  /* defined(__clang__) && __clang__ < 11 */
/// \ingroup PeoGroup
/// Clang's optimizer has a known bug that sometimes produces spurious
/// FPEs.  This indicates that the variable `var` can trigger that bug
/// and prevents some optimizations.  This is fixed upstream in Clang
/// 11.
#define VARIABLE_CAUSES_CLANG_FPE(var) ((void)(var))
#endif  /* defined(__clang__) && __clang__ < 11 */
