// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/AbortWithErrorMessage.hpp"

#include <boost/stacktrace.hpp>
#include <charm++.h>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "Utilities/ErrorHandling/Breakpoint.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/ErrorHandling/FormatStacktrace.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace {
template <bool ShowTrace, typename ExceptionTypeToThrow>
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
     << abbreviated_symbol_name(std::string{pretty_function}) << " in " << file
     << ":" << line << "\n"
     << "\n"
     << message << "\n";
  print_captures_for_error(os);
  os << "############ ERROR ############\n"
     << "\n";
  breakpoint();
  throw ExceptionTypeToThrow(os.str());
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
     << abbreviated_symbol_name(std::string{pretty_function}) << " in " << file
     << ":" << line << "\n"
     << "\n"
     << "'" << expression << "' violated!\n"
     << message << "\n";
  print_captures_for_error(os);
  os << "############ ASSERT FAILED ############\n"
     << "\n";
  breakpoint();
  throw SpectreAssert(os.str());
}

template <typename ExceptionTypeToThrow>
void abort_with_error_message(const char* file, const int line,
                              const char* pretty_function,
                              const std::string& message) {
  abort_with_error_message_impl<true, ExceptionTypeToThrow>(
      file, line, pretty_function, message);
}

void abort_with_error_message_no_trace(const char* file, const int line,
                                       const char* pretty_function,
                                       const std::string& message) {
  abort_with_error_message_impl<false, SpectreError>(file, line,
                                                     pretty_function, message);
}

#define GET_EX_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                       \
  template void abort_with_error_message<GET_EX_TYPE(data)>(         \
      const char* file, const int line, const char* pretty_function, \
      const std::string& message);

// STL exceptions first, group as on
// https://en.cppreference.com/w/cpp/error/exception
//
// Then SpECTRE exception classes.
GENERATE_INSTANTIATIONS(
    INSTANTIATION, (std::logic_error, std::invalid_argument, std::domain_error,
                    std::length_error, std::out_of_range,

                    std::runtime_error, std::range_error, std::overflow_error,
                    std::underflow_error,

                    std::ios_base::failure,

                    SpectreError, SpectreAssert, SpectreFpe, convergence_error))

#undef INSTANTIATION
#undef GET_EX_TYPE
