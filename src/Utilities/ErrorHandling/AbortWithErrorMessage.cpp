// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/AbortWithErrorMessage.hpp"

#include <boost/stacktrace.hpp>
#include <charm++.h>
#include <cstdlib>
#include <memory>
#include <sstream>

#include "Utilities/ErrorHandling/Breakpoint.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/ErrorHandling/FormatStacktrace.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace {
template <bool ShowTrace>
[[noreturn]] void abort_with_error_message_impl(const char* file,
                                                const int line,
                                                const char* pretty_function,
                                                const std::string& message) {
  std::ostringstream os;
  os << "\n"
     << "############ ERROR ############\n";
  if constexpr (ShowTrace) {
    os << "Stack trace:\n\n" << boost::stacktrace::stacktrace() << "\n";
  }
  os << "Wall time: " << sys::pretty_wall_time() << "\n"
     << "Node: " << sys::my_node() << " Proc: " << sys::my_proc() << "\n"
     << "Line: " << line << " of " << file << "\n"
     << "Function: " << pretty_function << "\n"
     << message << "\n"
     << "############ ERROR ############\n"
     << "\n";
  breakpoint();
  throw SpectreError(os.str());
}
}  // namespace

void abort_with_error_message(const char* expression, const char* file,
                              const int line, const char* pretty_function,
                              const std::string& message) {
  std::ostringstream os;
  os << "\n"
     << "############ ASSERT FAILED ############\n"
     << "Stack trace:\n\n"
     << boost::stacktrace::stacktrace() << "\n"
     << "Wall time: " << sys::pretty_wall_time() << "\n"
     << "Node: " << sys::my_node() << " Proc: " << sys::my_proc() << "\n"
     << "Line: " << line << " of " << file << "\n"
     << "'" << expression << "' violated!\n"
     << "Function: " << pretty_function << "\n"
     << message << "\n"
     << "############ ASSERT FAILED ############\n"
     << "\n";
  breakpoint();
  throw SpectreAssert(os.str());
}


void abort_with_error_message(const char* file, const int line,
                              const char* pretty_function,
                              const std::string& message) {
  abort_with_error_message_impl<true>(file, line, pretty_function, message);
}

void abort_with_error_message_no_trace(const char* file, const int line,
                                       const char* pretty_function,
                                       const std::string& message) {
  abort_with_error_message_impl<false>(file, line, pretty_function, message);
}
