// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \ingroup PeoGroup
/// Workarounds for optimizer bugs
namespace optimizer_hacks {
/// \ingroup PeoGroup
/// Produce a situation where the value of `*variable` could have been
/// changed without the optimizer's knowledge, without actually
/// changing the variable.
void indicate_value_can_be_changed(void* variable) noexcept;
}  // namespace optimizer_hacks

#ifdef __clang__
#define VARIABLE_CAUSES_CLANG_FPE(var) \
  ::optimizer_hacks::indicate_value_can_be_changed(&(var))
#else  /* __clang__ */
/// \ingroup PeoGroup
/// Clang's optimizer has a known bug that sometimes produces spurious
/// FPEs.  This indicates that the variable `var` can trigger that bug
/// and prevents some optimizations.  (See [the LLVM bug
/// report](https://llvm.org/bugs/show_bug.cgi?id=18673), which is
/// effectively WONTFIX.)
#define VARIABLE_CAUSES_CLANG_FPE(var) ((void)(var))
#endif  /* __clang__ */
