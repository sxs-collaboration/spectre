// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>
#include <vector>

namespace CaptureForError_detail {
class CaptureForErrorBase {
 protected:
  CaptureForErrorBase() = default;
  ~CaptureForErrorBase() = default;

 public:
  CaptureForErrorBase(CaptureForErrorBase&&) = delete;
  CaptureForErrorBase(const CaptureForErrorBase&) = delete;
  CaptureForErrorBase& operator=(CaptureForErrorBase&&) = delete;
  CaptureForErrorBase& operator=(const CaptureForErrorBase&) = delete;

  virtual void print(std::ostream& stream) const = 0;
};

class CaptureList {
 public:
  static CaptureList& the_list();

  void register_self(const CaptureForErrorBase* capture);
  void deregister_self(const CaptureForErrorBase* capture);

  void print(std::ostream& stream);

  CaptureList(CaptureList&&) = delete;
  CaptureList(const CaptureList&) = delete;
  CaptureList& operator=(CaptureList&&) = delete;
  CaptureList& operator=(const CaptureList&) = delete;

 private:
  CaptureList();
  ~CaptureList() noexcept(false);

  std::vector<const CaptureForErrorBase*> captures_{};
  bool currently_printing_ = false;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
template <typename T>
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class CaptureForError : public CaptureForErrorBase {
 public:
  CaptureForError() = delete;
  CaptureForError(CaptureForError&&) = delete;
  CaptureForError(const CaptureForError&) = delete;
  CaptureForError& operator=(CaptureForError&&) = delete;
  CaptureForError& operator=(const CaptureForError&) = delete;

  CaptureForError(const char* const name, const T& capture)
      : name_(name), capture_(capture) {
    CaptureList::the_list().register_self(this);
  }

  // Capturing an rvalue would lead to a dangling reference.
  CaptureForError(const char* /*name*/, const T&& /*capture*/) = delete;

  ~CaptureForError() noexcept(false) {
    CaptureList::the_list().deregister_self(this);
  }

  void print(std::ostream& stream) const override {
    stream << name_ << " = " << capture_ << "\n";
  }

 private:
  const char* name_;
  const T& capture_;
};
#pragma GCC diagnostic pop
}  // namespace CaptureForError_detail

/// \cond
#define CAPTURE_FOR_ERROR_NAME2(line) capture_for_error_impl##line
#define CAPTURE_FOR_ERROR_NAME(line) CAPTURE_FOR_ERROR_NAME2(line)
/// \endcond

/*!
 * \brief Capture a variable to be printed on ERROR or ASSERT.
 *
 * The argument will be printed using its stream operator if an error
 * occurs before the end of the current scope.  The object is captured
 * by reference, so it must live until the end of the current scope,
 * and any subsequent changes to it will be reflected in the error
 * message.
 */
#define CAPTURE_FOR_ERROR(var)                                          \
  const CaptureForError_detail::CaptureForError CAPTURE_FOR_ERROR_NAME( \
      __LINE__)(#var, var)

/*!
 * \brief Stream all objects currently captured by CaptureForError.
 */
void print_captures_for_error(std::ostream& stream);
