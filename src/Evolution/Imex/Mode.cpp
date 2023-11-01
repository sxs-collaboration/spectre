// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Imex/Mode.hpp"

#include <string>

#include "Options/ParseOptions.hpp"

template <>
imex::Mode Options::create_from_yaml<imex::Mode>::create<void>(
    const Options::Option& options) {
  const auto mode = options.parse_as<std::string>();
  if (mode == "Implicit") {
    return imex::Mode::Implicit;
  } else if (mode == "SemiImplicit") {
    return imex::Mode::SemiImplicit;
  } else {
    PARSE_ERROR(options.context(),
                "Invalid IMEX mode.  Must be Implicit or SemiImplicit.");
  }
}
