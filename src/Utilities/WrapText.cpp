// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/WrapText.hpp"

#include <cstddef>
#include <string>

#include "Utilities/ErrorHandling/Assert.hpp"

std::string wrap_text(std::string str, const size_t line_length,
                      const std::string& indentation) noexcept {
  str = indentation + str;
  ASSERT(indentation.size() < line_length,
         "The indentation must be shorter than the line length. Indentation "
         "length is: "
             << indentation.size() << " while line length is: " << line_length);

  // Insert indentation at newlines in the string first to ensure all new lines
  // are indented correctly.
  if (not indentation.empty()) {
    size_t newline_location = str.find('\n');
    while (newline_location != std::string::npos) {
      str.insert(newline_location + 1, indentation);
      newline_location = str.find('\n', newline_location + 1);
    }
  }

  // Wrap the string to the set length of characters
  for (size_t i = 0; i + line_length < str.size();) {
    // Find the last newline, and split there if it is within the next
    // line_length characters.
    if (const size_t last_newline = str.rfind('\n', i + line_length);
        last_newline != std::string::npos and last_newline > i) {
      i = last_newline + 1;
    } else if (const size_t last_space = str.rfind(' ', i + line_length);
               last_space <= i + indentation.size() or
               last_space == std::string::npos or
               (i == 0 and str.substr(i, last_space + 1) ==
                               std::string(last_space + 1, ' '))) {
      // The last 'or' condition of the `if` above handles the edge case where
      // the first space in the first line is precedes only spaces.
      const size_t insert_location = i + line_length - 1;
      str.insert(insert_location, "-\n" + indentation);
      i = insert_location + 2;
    } else {
      str.at(last_space) = '\n';
      str.insert(last_space + 1, indentation);
      i = last_space + 1;
    }
  }
  return str;
}
