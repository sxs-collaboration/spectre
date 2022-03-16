// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Check whether it is safe to throw an exception from a destructor.
 *
 * This class is only useful when used as a member variable.  In that
 * case, its call operator will return `true` if an exception can be
 * thrown without immediately calling std::terminate because of stack
 * unwinding.  It will always return true if not called during the
 * execution of the class's destructor.
 *
 * Remember that any class wishing to throw an exception from it's
 * destructor must explicitly mark the destructor `noexcept(false)`.
 */
class CanSafelyThrow {
 public:
  CanSafelyThrow();
  CanSafelyThrow(const CanSafelyThrow&);
  CanSafelyThrow(CanSafelyThrow&&);
  CanSafelyThrow& operator=(const CanSafelyThrow&);
  CanSafelyThrow& operator=(CanSafelyThrow&&);

  bool operator()() const noexcept;

 private:
  int uncaught_exceptions_;
};
