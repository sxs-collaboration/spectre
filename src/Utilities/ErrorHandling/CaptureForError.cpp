// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/CaptureForError.hpp"

#include <ostream>
#include <vector>

#include "Utilities/ErrorHandling/Assert.hpp"

namespace CaptureForError_detail {
CaptureList& CaptureList::the_list() {
  thread_local CaptureList singleton{};
  return singleton;
}

void CaptureList::register_self(const CaptureForErrorBase* const capture) {
  captures_.push_back(capture);
}

void CaptureList::deregister_self(const CaptureForErrorBase* const capture) {
  ASSERT(not captures_.empty() and captures_.back() == capture,
         "Deregistering wrong capture.");
  captures_.pop_back();
}

void CaptureList::print(std::ostream& stream) {
  if (currently_printing_) {
    // Error during an error.  Don't recurse.
    return;
  }
  try {
    currently_printing_ = true;
    if (not captures_.empty()) {
      stream << "\nCaptured variables:\n";
    }
    for (const auto& capture : captures_) {
      capture->print(stream);
    }
  } catch (...) {
    currently_printing_ = false;
    throw;
  }
  currently_printing_ = false;
}

CaptureList::CaptureList() = default;

CaptureList::~CaptureList() noexcept(false) {
  ASSERT(captures_.empty(), "Not all captures deregistered.");
}
}  // namespace CaptureForError_detail

void print_captures_for_error(std::ostream& stream) {
  CaptureForError_detail::CaptureList::the_list().print(stream);
}
