// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"

namespace fd {
std::ostream& operator<<(std::ostream& os, DerivativeOrder der_order) {
  switch (der_order) {
    case DerivativeOrder::OneHigherThanRecons:
      return os << "OneHigherThanRecons";
    case DerivativeOrder::OneHigherThanReconsButFiveToFour:
      return os << "OneHigherThanReconsButFiveToFour";
    case DerivativeOrder::Two:
      return os << "2";
    case DerivativeOrder::Four:
      return os << "4";
    case DerivativeOrder::Six:
      return os << "6";
    case DerivativeOrder::Eight:
      return os << "8";
    case DerivativeOrder::Ten:
      return os << "10";
    default:
      ERROR("Unknown value for DerivativeOrder");
  };
}
}  // namespace fd

template <>
fd::DerivativeOrder
Options::create_from_yaml<fd::DerivativeOrder>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if (type_read == get_output(fd::DerivativeOrder::OneHigherThanRecons)) {
    return fd::DerivativeOrder::OneHigherThanRecons;
  } else if (type_read ==
             get_output(
                 fd::DerivativeOrder::OneHigherThanReconsButFiveToFour)) {
    return fd::DerivativeOrder::OneHigherThanReconsButFiveToFour;
  } else if (type_read == get_output(fd::DerivativeOrder::Two)) {
    return fd::DerivativeOrder::Two;
  } else if (type_read == get_output(fd::DerivativeOrder::Four)) {
    return fd::DerivativeOrder::Four;
  } else if (type_read == get_output(fd::DerivativeOrder::Six)) {
    return fd::DerivativeOrder::Six;
  } else if (type_read == get_output(fd::DerivativeOrder::Eight)) {
    return fd::DerivativeOrder::Eight;
  } else if (type_read == get_output(fd::DerivativeOrder::Ten)) {
    return fd::DerivativeOrder::Ten;
  }
  PARSE_ERROR(
      options.context(),
      "Failed to convert \""
          << type_read << "\" to DerivativeOrder. Must be one of '"
          << get_output(fd::DerivativeOrder::OneHigherThanRecons) << "', '"
          << get_output(fd::DerivativeOrder::OneHigherThanReconsButFiveToFour)
          << "', 2, 4, 6, 8, or 10.");
}
