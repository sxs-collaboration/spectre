// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/CanSafelyThrow.hpp"

#include <exception>

CanSafelyThrow::CanSafelyThrow()
    : uncaught_exceptions_(std::uncaught_exceptions()) {}

// None of these operations change when the destructor is run, which
// is the important thing for this class.
CanSafelyThrow::CanSafelyThrow(const CanSafelyThrow&) : CanSafelyThrow() {}
CanSafelyThrow::CanSafelyThrow(CanSafelyThrow&&) : CanSafelyThrow() {}
CanSafelyThrow& CanSafelyThrow::operator=(const CanSafelyThrow&) {
  return *this;
}
CanSafelyThrow& CanSafelyThrow::operator=(CanSafelyThrow&&) { return *this; }

bool CanSafelyThrow::operator()() const noexcept {
  return uncaught_exceptions_ == std::uncaught_exceptions();
}
