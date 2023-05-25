// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

#include <csignal>

#if SPECTRE_FPE_CSR
#include <xmmintrin.h>
#elif SPECTRE_FPE_FENV
#include <cfenv>
#endif

namespace {

#if SPECTRE_FPE_CSR
const unsigned int disabled_mask = _mm_getcsr();
const unsigned int exception_flags =
    _MM_MASK_MASK & ~(_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO);
#elif SPECTRE_FPE_FENV
const int exception_flags = FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW;
#endif

[[noreturn]] void fpe_signal_handler(int /*signal*/) {
  ERROR("Floating point exception!");
}
}  // namespace

void enable_floating_point_exceptions() {
#if SPECTRE_FPE_CSR
  _mm_setcsr(exception_flags);
#elif SPECTRE_FPE_FENV
  feenableexcept(exception_flags);
#endif
  std::signal(SIGFPE, fpe_signal_handler);
}

void disable_floating_point_exceptions() {
#if SPECTRE_FPE_CSR
  _mm_setcsr(disabled_mask);
#elif SPECTRE_FPE_FENV
  fedisableexcept(exception_flags);
#endif
}

ScopedFpeState::~ScopedFpeState() { restore_exceptions(); }

ScopedFpeState::ScopedFpeState() { save_exceptions(); }

ScopedFpeState::ScopedFpeState(const bool exceptions_enabled)
    : ScopedFpeState() {
  set_exceptions(exceptions_enabled);
}

ScopedFpeState::ScopedFpeState(ScopedFpeState::DoNotSave /*meta*/) {}

void ScopedFpeState::set_exceptions(const bool exceptions_enabled) const {
  ASSERT(original_state_.has_value(), "FPE state not saved.");
  if (exceptions_enabled) {
    enable_floating_point_exceptions();
  } else {
    disable_floating_point_exceptions();
  }
}

void ScopedFpeState::save_exceptions() {
  ASSERT(not original_state_.has_value(), "FPE state already saved.");
#if SPECTRE_FPE_CSR
  original_state_.emplace(_mm_getcsr());
#elif SPECTRE_FPE_FENV
  original_state_.emplace(fegetexcept());
#else
  original_state_.emplace();
#endif
}

void ScopedFpeState::restore_exceptions() {
  if (not original_state_.has_value()) {
    return;
  }
#if SPECTRE_FPE_CSR
  _mm_setcsr(*original_state_);
#elif SPECTRE_FPE_FENV
  fedisableexcept(exception_flags);
  feenableexcept(*original_state_);
#endif
  original_state_.reset();
}
