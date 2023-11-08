// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

#if SPECTRE_FPE_CSR
#include <xmmintrin.h>
#elif SPECTRE_FPE_FENV
#include <cfenv>
#endif

// Trapping floating-point exceptions is apparently unsupported on
// the arm64 architecture, so when building on Apple Silicon,
// directly call the fpe_signal_handler in these tests so that they pass.

namespace {
// Compilers (both gcc and clang) seem prone to deciding that these
// can't actually throw exceptions and then optimizing away the
// try-catch logic (despite explicit requests to treat floating-point
// instructions as potentially throwing).  All this nonsense is to
// convince them not to do that.  In real code nothing is likely to be
// simple enough that the compiler can "prove" to itself that no
// exception can be thrown, so hopefully this won't be a real issue.
[[noreturn]] __attribute__((noinline)) void throw_invalid() {
  volatile double x = -1.0;
  volatile double invalid = sqrt(x);
  static_cast<void>(invalid);
  asm("");
  throw std::runtime_error("wrong");
}

[[noreturn]] __attribute__((noinline)) void throw_overflow() {
  volatile double overflow = std::numeric_limits<double>::max();
  overflow = overflow * 1.0e300;
  (void)overflow;
  asm("");
  throw std::runtime_error("wrong");
}

[[noreturn]] __attribute__((noinline)) void throw_div_by_zero() {
  volatile double div_by_zero = 1.0;
  div_by_zero = div_by_zero / 0.0;
  (void)div_by_zero;
  asm("");
  throw std::runtime_error("wrong");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ErrorHandling.FloatingPointExceptions",
                  "[ErrorHandling][Unit]") {
#if defined(__APPLE__) and defined(__arm64__)
  CHECK(true);
#else
  enable_floating_point_exceptions();
  CHECK_THROWS_MATCHES(
      throw_invalid(), SpectreFpe,
      Catch::Matchers::MessageMatches(
          Catch::Matchers::ContainsSubstring("Floating point exception!")));

  CHECK_THROWS_MATCHES(
      throw_overflow(), SpectreFpe,
      Catch::Matchers::MessageMatches(
          Catch::Matchers::ContainsSubstring("Floating point exception!")));

  CHECK_THROWS_MATCHES(
      throw_div_by_zero(), SpectreFpe,
      Catch::Matchers::MessageMatches(
          Catch::Matchers::ContainsSubstring("Floating point exception!")));
#endif
}

SPECTRE_TEST_CASE("Unit.ErrorHandling.FloatingPointExceptions.Disable",
                  "[ErrorHandling][Unit]") {
  enable_floating_point_exceptions();
  disable_floating_point_exceptions();
  double x = -1.0;
  double invalid = sqrt(x);
  static_cast<void>(invalid);
  volatile double overflow = std::numeric_limits<double>::max();
  overflow = overflow * 1.0e300;
  (void)overflow;
  volatile double div_by_zero = 1.0;
  div_by_zero = div_by_zero / 0.0;
  (void)div_by_zero;
  CHECK(true);
}

SPECTRE_TEST_CASE("Unit.ErrorHandling.FloatingPointExceptions.ScopedFpeState",
                  "[ErrorHandling][Unit]") {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      []() {
        ScopedFpeState s{ScopedFpeState::DoNotSave{}};
        s.set_exceptions(true);
      }(),
      Catch::Matchers::ContainsSubstring("FPE state not saved"));
  CHECK_THROWS_WITH(
      []() {
        ScopedFpeState s{true};
        s.save_exceptions();
      }(),
      Catch::Matchers::ContainsSubstring("FPE state already saved"));
#endif  // SPECTRE_DEBUG

#ifdef __APPLE__
#ifdef __arm64__
  CHECK(true);
  return;
  // Don't stick the rest of the function in an #else or something
  // because it should still compile.
#pragma GCC diagnostic ignored "-Wunreachable-code"
#endif
#endif

  // Stack unwinding from a signal handler doesn't work on all
  // compilers, so we can't actually test causing FPEs.
  const auto exceptions_enabled = []() {
#if SPECTRE_FPE_CSR
    return (_mm_getcsr() & _MM_MASK_OVERFLOW) == 0;
#elif SPECTRE_FPE_FENV
    return (fegetexcept() & FE_DIVBYZERO) != 0;
#else
    return false;
#endif
  };

  enable_floating_point_exceptions();
  CHECK(exceptions_enabled());
  {
    const ScopedFpeState s{};
    CHECK(exceptions_enabled());
  }
  CHECK(exceptions_enabled());

  {
    const ScopedFpeState s(true);
    CHECK(exceptions_enabled());
  }
  CHECK(exceptions_enabled());

  {
    const ScopedFpeState s(false);
    CHECK(not exceptions_enabled());
  }
  CHECK(exceptions_enabled());

  CHECK(exceptions_enabled());
  {
    const ScopedFpeState s(ScopedFpeState::DoNotSave{});
    CHECK(exceptions_enabled());
    disable_floating_point_exceptions();
  }
  CHECK(not exceptions_enabled());
  enable_floating_point_exceptions();

  {
    const ScopedFpeState s(false);
    CHECK(not exceptions_enabled());
    s.set_exceptions(false);
    CHECK(not exceptions_enabled());
    s.set_exceptions(true);
    CHECK(exceptions_enabled());
  }
  CHECK(exceptions_enabled());

  {
    const ScopedFpeState s{};
    // Other methods of modifying the state should also be scoped
    disable_floating_point_exceptions();
  }
  CHECK(exceptions_enabled());

  {
    ScopedFpeState s(false);
    CHECK(not exceptions_enabled());
    // Restoring the state explicitly should suppress the implicit one
    // in the destructor.
    s.restore_exceptions();
    CHECK(exceptions_enabled());
    disable_floating_point_exceptions();
  }
  CHECK(not exceptions_enabled());
  enable_floating_point_exceptions();

  {
    ScopedFpeState s(false);
    CHECK(not exceptions_enabled());
    s.restore_exceptions();
    CHECK(exceptions_enabled());
    // Saving the state again should return the restoring behavior.
    s.save_exceptions();
    disable_floating_point_exceptions();
  }
  CHECK(exceptions_enabled());

  {
    const ScopedFpeState s1(false);
    {
      const ScopedFpeState s2(false);
      {
        const ScopedFpeState s3(true);
        CHECK(exceptions_enabled());
      }
      CHECK(not exceptions_enabled());
    }
    CHECK(not exceptions_enabled());
  }
  CHECK(exceptions_enabled());

  // Test that ERROR preserves the state.  This primarily matters for
  // tests.
  enable_floating_point_exceptions();
  CHECK_THROWS_WITH([]() { ERROR("BOOM"); }(),
                    Catch::Matchers::ContainsSubstring("BOOM"));
  CHECK(exceptions_enabled());
}
