// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <exception>
#include <string>

#include "Utilities/ErrorHandling/CanSafelyThrow.hpp"

namespace {
class TestThrowingError : public std::runtime_error {
 public:
  TestThrowingError(const char* what) : std::runtime_error(what) {}
  // Catch wants this.
  operator std::string() const { return what(); }
};

class TestThrowing {
 public:
  TestThrowing(const bool should_be_safe, const char* const message = nullptr)
      : should_be_safe_(should_be_safe), message_(message) {}

  void reset(const bool should_be_safe, const char* const message = nullptr) {
    should_be_safe_ = should_be_safe;
    message_ = message;
  }

  ~TestThrowing() noexcept(false) {
    CHECK(can_safely_throw_() == should_be_safe_);
    if (should_be_safe_ and message_) {
      throw TestThrowingError(message_);
    }
  }

 private:
  CanSafelyThrow can_safely_throw_{};
  bool should_be_safe_{};
  // const char* instead of std::string for simpler move semantics.
  const char* message_ = "";
};

class TestThrowingNested {
 public:
  TestThrowingNested(const bool unwinding)
      : unwinding_(unwinding),
        thrower_(not unwinding_),
        copy_(not unwinding_),
        move_(not unwinding_),
        copy_from_inner_(not unwinding_),
        move_from_inner_(not unwinding_),
        copy_from_inner2_(not unwinding_),
        move_from_inner2_(not unwinding_) {}

  ~TestThrowingNested() noexcept(false) {
    copy_ = thrower_;
    copy_.reset(not unwinding_);
    move_ = std::move(thrower_);
    move_.reset(not unwinding_);

    try {
      TestThrowing inner_thrower(true);
      copy_from_inner_ = inner_thrower;
      copy_from_inner_.reset(not unwinding_);
      move_from_inner_ = std::move(inner_thrower);
      move_from_inner_.reset(not unwinding_);
      auto copied_from_outer = thrower_;
      copied_from_outer.reset(true);
      auto moved_from_outer = std::move(thrower_);
      moved_from_outer.reset(true);
    } catch (int) {
    }

    try {
      TestThrowing inner_thrower2(false);
      copy_from_inner2_ = inner_thrower2;
      copy_from_inner2_.reset(not unwinding_);
      move_from_inner2_ = std::move(inner_thrower2);
      move_from_inner2_.reset(not unwinding_);
      auto copied_from_outer = thrower_;
      copied_from_outer.reset(false);
      auto moved_from_outer = std::move(thrower_);
      moved_from_outer.reset(false);
      throw int{};
    } catch (int) {
    }
  }

 private:
  bool unwinding_;
  TestThrowing thrower_;
  TestThrowing copy_;
  TestThrowing move_;
  TestThrowing copy_from_inner_;
  TestThrowing move_from_inner_;
  TestThrowing copy_from_inner2_;
  TestThrowing move_from_inner2_;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.ErrorHandling.CanSafelyThrow",
                  "[Unit][Utilities][ErrorHandling]") {
  CHECK_THROWS_MATCHES([]() { TestThrowing thrower(true, "From thrower"); }(),
                       TestThrowingError, Catch::Contains("From thrower"));

  CHECK_THROWS_MATCHES(
      []() {
        TestThrowing thrower(false);
        throw TestThrowingError("From elsewhere");
      }(),
      TestThrowingError, Catch::Contains("From elsewhere"));

  CHECK_THROWS_MATCHES(
      []() {
        try {
          throw int{};
        } catch (int) {
          TestThrowing thrower(true, "From thrower");
        }
      }(),
      TestThrowingError, Catch::Contains("From thrower"));

  { TestThrowingNested(false); }

  try {
    TestThrowingNested test(true);
    throw int{};
  } catch (int) {
  }
}
