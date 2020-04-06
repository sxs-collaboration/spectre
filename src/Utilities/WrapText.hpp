// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

/// \ingroup UtilitiesGroup
/// \brief Wrap the string `str` so that it is no longer than `line_length` and
/// indent each new line with `indentation`. The first line is also indented.
///
/// Single words longer than `line_length` are hyphenated.
std::string wrap_text(std::string str, size_t line_length,
                      const std::string& indentation = "") noexcept;
