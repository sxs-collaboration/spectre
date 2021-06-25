// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/FloatingPointType.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"

std::ostream& operator<<(std::ostream& os,
                         const FloatingPointType& t) noexcept {
  switch (t) {
    case FloatingPointType::Float:
      return os << "Float";
    case FloatingPointType::Double:
      return os << "Double";
    default:
      ERROR("Unknown floating point type, must be Float or Double");
  }
}

template <>
FloatingPointType Options::create_from_yaml<FloatingPointType>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if (type_read == get_output(FloatingPointType::Float)) {
    return FloatingPointType::Float;
  } else if (type_read == get_output(FloatingPointType::Double)) {
    return FloatingPointType::Double;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read << "\" to FloatingPointType. Must be one of '"
                  << get_output(FloatingPointType::Float) << "' or '"
                  << get_output(FloatingPointType::Double) << "'.");
}
