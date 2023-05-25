// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Functions to enable/disable termination on floating point exceptions

#pragma once

#include <optional>

/// \cond
#ifdef __APPLE__
#ifndef __arm64__
#define SPECTRE_FPE_CSR 1
#endif
#else
#define SPECTRE_FPE_FENV 1
#endif
/// \endcond

/// \ingroup ErrorHandlingGroup
/// After a call to this function, the code will terminate with a floating
/// point exception on overflow, divide-by-zero, and invalid operations.
void enable_floating_point_exceptions();

/// \ingroup ErrorHandlingGroup
/// After a call to this function, the code will NOT terminate with a floating
/// point exception on overflow, divide-by-zero, and invalid operations.
///
/// \warning Do not use this function to temporarily disable FPEs,
/// because it will not interact correctly with C++ exceptions.  Use
/// `ScopedFpeState` instead.
void disable_floating_point_exceptions();

/// \ingroup ErrorHandlingGroup
/// An RAII object to temporarily modify the handling of floating
/// point exceptions.
class ScopedFpeState {
 public:
  ScopedFpeState(const ScopedFpeState&) = delete;
  ScopedFpeState(ScopedFpeState&&) = delete;
  ScopedFpeState& operator=(const ScopedFpeState&) = delete;
  ScopedFpeState& operator=(ScopedFpeState&&) = delete;
  ~ScopedFpeState();

  /// Start a scope that will be restored, without changing the
  /// current state.
  ScopedFpeState();
  /// Start a scope with the specified exception state.  This is
  /// equivalent to calling the default constructor followed by
  /// `set_exceptions`.
  explicit ScopedFpeState(bool exceptions_enabled);

  struct DoNotSave {};
  /// Start a scope without saving the current state.  The only valid
  /// method call from this state is `save_exceptions`.
  explicit ScopedFpeState(DoNotSave /*meta*/);

  /// Enable or disable floating point exceptions.  It is an error if
  /// the exception state is not currently saved.
  void set_exceptions(bool exceptions_enabled) const;

  /// Save the current exception handling state after it has been
  /// cleared by `restore_exceptions`.  It will be restored by a later
  /// call to `restore_exceptions`.  It is an error to call this if
  /// a state is already saved.
  void save_exceptions();

  /// Restore the FPE handling to the internally saved state if
  /// present and clear that state.  This is called automatically by
  /// the destructor if it is not called manually.
  void restore_exceptions();

 private:
#if SPECTRE_FPE_CSR
  std::optional<unsigned int> original_state_;
#elif SPECTRE_FPE_FENV
  std::optional<int> original_state_;
#else
  // FPEs not supported, but this is still used to check method calls
  // are valid.
  struct DummyState {};
  std::optional<DummyState> original_state_;
#endif
};
