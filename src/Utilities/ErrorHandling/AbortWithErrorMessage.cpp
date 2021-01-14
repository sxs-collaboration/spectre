// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/AbortWithErrorMessage.hpp"

#include <array>
#include <charm++.h>
#include <execinfo.h>
#include <memory>
#include <sstream>

#include "Utilities/ErrorHandling/Breakpoint.hpp"
#include "Utilities/System/Abort.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace {
struct FillBacktrace {};

std::ostream& operator<<(std::ostream& os, const FillBacktrace& /*unused*/) {
  // 3 for the stream operator and abort_with_error_message, 10 for the
  // stack.
  constexpr size_t max_stack_depth_printed = 13;
  std::array<void*, max_stack_depth_printed> trace_elems{};
  int trace_elem_count = backtrace(trace_elems.data(), max_stack_depth_printed);
  std::unique_ptr<char*, decltype(free)*> stack_syms{
      backtrace_symbols(trace_elems.data(), trace_elem_count), free};
  // Start at 3 to ignore stream operator and abort_with_error_message
  for (int i = 3; i < trace_elem_count; ++i) {
    os << stack_syms.get()[i] << "\n";
  }
  return os;
}

template <bool ShowTrace>
[[noreturn]] void abort_with_error_message_impl(const char* file,
                                                const int line,
                                                const char* pretty_function,
                                                const std::string& message) {
  std::ostringstream os;
  os << "\n"
     << "############ ERROR ############\n";
  if constexpr (ShowTrace) {
    os << "Shortened stack trace is:\n"
       << FillBacktrace{} << "End shortened stack trace.\n\n";
  }
  os << "Node: " << sys::my_node() << " Proc: " << sys::my_proc() << "\n"
     << "Line: " << line << " of " << file << "\n"
     << "Function: " << pretty_function << "\n"
     << message << "\n"
     << "############ ERROR ############\n"
     << "\n";
  // We use CkError instead of abort to print the error message because in the
  // case of an executable not using Charm++'s main function the call to abort
  // will segfault before anything is printed.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  CkError("%s", os.str().c_str());
  breakpoint();
  sys::abort("");
}
}  // namespace

void abort_with_error_message(const char* expression, const char* file,
                              const int line, const char* pretty_function,
                              const std::string& message) {
  std::ostringstream os;
  os << "\n"
     << "############ ASSERT FAILED ############\n"
     << "Shortened stack trace is:\n"
     << FillBacktrace{} << "End shortened stack trace.\n\n"
     << "Node: " << sys::my_node() << " Proc: " << sys::my_proc() << "\n"
     << "Line: " << line << " of " << file << "\n"
     << "'" << expression << "' violated!\n"
     << "Function: " << pretty_function << "\n"
     << message << "\n"
     << "############ ASSERT FAILED ############\n"
     << "\n";
  // We use CkError instead of abort to print the error message because in the
  // case of an executable not using Charm++'s main function the call to abort
  // will segfault before anything is printed.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  CkError("%s", os.str().c_str());
  breakpoint();
  sys::abort("");
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
