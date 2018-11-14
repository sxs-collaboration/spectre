// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ErrorHandling/AbortWithErrorMessage.hpp"

#include <sstream>

#include "Parallel/Abort.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Printf.hpp"

void abort_with_error_message(const char* expression, const char* file,
                              const int line, const char* pretty_function,
                              const std::string& message) {
  std::ostringstream os;
  os << "\n"
     << "############ ASSERT FAILED ############\n"
     << "Node: " << Parallel::my_node() << " Proc: " << Parallel::my_proc()
     << "\n"
     << "Line: " << line << " of " << file << "\n"
     << "'" << expression << "' violated!\n"
     << "Function: " << pretty_function << "\n"
     << message << "\n"
     << "############ ASSERT FAILED ############\n"
     << "\n";
  // We use printf instead of abort to print the error message because in the
  // case of an executable not using Charm++'s main function the call to abort
  // will segfault before anything is printed.
  Parallel::printf_error(os.str());
  Parallel::abort("");
}

void abort_with_error_message(const char* file, const int line,
                              const char* pretty_function,
                              const std::string& message) {
  std::ostringstream os;
  os << "\n"
     << "############ ERROR ############\n"
     << "Node: " << Parallel::my_node() << " Proc: " << Parallel::my_proc()
     << "\n"
     << "Line: " << line << " of " << file << "\n"
     << "Function: " << pretty_function << "\n"
     << message << "\n"
     << "############ ERROR ############\n"
     << "\n";
  // We use printf instead of abort to print the error message because in the
  // case of an executable not using Charm++'s main function the call to abort
  // will segfault before anything is printed.
  Parallel::printf_error(os.str());
  Parallel::abort("");
}
