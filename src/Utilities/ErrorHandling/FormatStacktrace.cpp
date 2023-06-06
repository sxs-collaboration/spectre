// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/FormatStacktrace.hpp"

#include <array>
#include <boost/core/demangle.hpp>
#include <boost/stacktrace.hpp>
#include <cstddef>
#include <cstdlib>     // For std::getenv
#include <execinfo.h>  // For backtrace_symbols
// link.h is not available on all platforms
#if __has_include(<link.h>)
#include <link.h>
#endif
#include <memory>
#include <sstream>
#include <string>

namespace {

std::string abbreviated_symbol_name(const std::string& symbol_name) {
  // We display the first `abbrev_length_from_start` characters of the symbol
  // name, then " [...] ", and then the last `abbrev_length_from_end` characters
  // of the symbol name.
  constexpr size_t abbrev_length_from_start = 300;
  constexpr size_t abbrev_length_from_end = 100;
  // Length of " [...] "
  constexpr size_t abbrev_length_separator = 7;
  if (symbol_name.size() <= abbrev_length_from_start + abbrev_length_from_end +
                                abbrev_length_separator) {
    return symbol_name;
  }
  // Allow an environment variable to toggle the abbreviation
  const char* show_full_symbol_env =
      std::getenv("SPECTRE_SHOW_FULL_BACKTRACE_SYMBOLS");
  if (show_full_symbol_env != nullptr and
      not std::string{show_full_symbol_env}.empty()) {
    return symbol_name;
  }
  return symbol_name.substr(0, abbrev_length_from_start) + " [...] " +
         symbol_name.substr(symbol_name.size() - abbrev_length_from_end,
                            abbrev_length_from_end);
}

std::string get_stack_frame(void* addr) {
  std::unique_ptr<char*, decltype(free)*> stack_syms{
      backtrace_symbols(&addr, 1), free};
  return stack_syms.get()[0];
}

#if __has_include(<link.h>)
std::string addr2line_command(const void* addr) {
  std::stringstream ss;
  ss << "addr2line -fCpe ";
  // Convert the address to the virtual memory address inside the
  // library/executable. This is the address that addr2line and llvm-addr2line
  // expect.
  Dl_info info;
  link_map* link_map = nullptr;
  dladdr1(addr, &info,
          reinterpret_cast<void**>(&link_map),  // NOLINT
          RTLD_DL_LINKMAP);
  ss << info.dli_fname;
  ss << " ";
  // NOLINTNEXTLINE
  ss << reinterpret_cast<void*>(reinterpret_cast<size_t>(addr) -
                                link_map->l_addr);
  return ss.str();
}
#endif  // __has_include(<link.h>)

}  // namespace

std::ostream& operator<<(std::ostream& os,
                         const boost::stacktrace::stacktrace& backtrace) {
  const std::streamsize standard_width = os.width();
  const size_t frames = backtrace.size();
  for (size_t i = 0; i < frames; ++i) {
    const auto& frame = backtrace[i];
    const std::string& symbol_name = frame.name();
    const std::string& source_file = frame.source_file();
    const size_t source_line = frame.source_line();
    // Enumerate frame number
    os.width(3);
    os << i;
    os.width(standard_width);
    os << ". ";
    if (frame.empty()) {
      os << "[empty]";
      continue;
    }
    // Skip frames originating in error handling code
    if (symbol_name.find("abort_with_error_message") != std::string::npos or
        source_file.find("src/Utilities/ErrorHandling/") != std::string::npos or
        symbol_name.find("boost::stacktrace") != std::string::npos) {
      os << "[error handling]\n";
      continue;
    }
    if (symbol_name.empty()) {
      // Boost was unable to get the symbol name (probably because dladdr
      // can't find it either). Fall back to printing the default stack
      // frame.
      // NOLINTNEXTLINE
      os << get_stack_frame(const_cast<void*>(frame.address()));
    } else {
      // Print symbol name. Abbreviate if necessary to avoid filling the
      // screen with templates.
      os << abbreviated_symbol_name(symbol_name);
    }
    if (source_line != 0) {
      // Print location in source file if available
      os << " in " << source_file << ":" << source_line;
    } else {
#if __has_include(<link.h>)
      // Print addr2line information otherwise
      os << " - Resolve source file and line with: "
         << addr2line_command(frame.address());
#endif
    }
    os << '\n';
  }

  return os;
}
