// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>
#include <string>

namespace Options {
/// \ingroup OptionParsingGroup
/// Information about the nested operations being performed by the
/// parser, for use in printing errors.  A default-constructed
/// Context is printed as an empty string.  This struct is
/// primarily used as an argument to PARSE_ERROR for reporting input
/// file parsing errors.  Users outside of the core option parsing
/// code should not need to manipulate the contents.
struct Context {
  bool top_level{true};
  /// (Part of) the parsing "backtrace" printed with an error
  std::string context;
  /// File line number (0 based)
  int line{-1};
  /// File column number (0 based)
  int column{-1};

  /// Append a line to the context.  Automatically appends a colon.
  void append(const std::string& c) { context += c + ":\n"; }
};

std::ostream& operator<<(std::ostream& s, const Context& c);

}  // namespace Options
