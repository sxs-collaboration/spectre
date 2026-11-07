// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Utilities/ErrorHandling/DeviceMessage.hpp"

namespace {
enum class Color : uint8_t { Red = 2, Green = 5 };

template <typename... Ts>
std::string stream_all(const Ts&... values) {
  DeviceMessage message{};
  (message << ... << values);
  CHECK(message.size() == std::string{message.c_str()}.size());
  return message.c_str();
}

void test_types() {
  CHECK(stream_all("Hello", ' ', "World") == "Hello World");
  const char* const c_string = "abc";
  CHECK(stream_all(c_string) == "abc");
  CHECK(stream_all(static_cast<const char*>(nullptr)) == "(null)");
  CHECK(stream_all(true, false) == "10");
  CHECK(stream_all(0, -12, 345U) == "0-12345");
  CHECK(stream_all(std::numeric_limits<long long>::min()) ==
        "-9223372036854775808");
  CHECK(stream_all(std::numeric_limits<unsigned long long>::max()) ==
        "18446744073709551615");
  CHECK(stream_all(size_t{42}) == "42");
  CHECK(stream_all(Color::Green) == "5");
  CHECK(stream_all(std::array<int, 3>{{1, -2, 3}}) == "(1,-2,3)");
  CHECK(stream_all(std::array<double, 0>{}) == "()");
  CHECK(stream_all(std::array<std::array<int, 2>, 2>{{{{1, 2}}, {{3, 4}}}}) ==
        "((1,2),(3,4))");
  // Unsupported types
  CHECK(stream_all("a", std::vector<double>{1.0}, "b") == "a<?>b");
  CHECK(stream_all(std::string{"abc"}) == "<?>");
  // Manipulators
  using OstreamManipulator = std::ostream& (*)(std::ostream&);
  using IosManipulator = std::ios_base& (*)(std::ios_base&);
  CHECK(stream_all("a", static_cast<OstreamManipulator>(std::endl), "b") ==
        "a\nb");
  CHECK(stream_all(std::setprecision(3),
                   static_cast<IosManipulator>(std::scientific), std::setw(5),
                   std::setfill('x'),
                   static_cast<IosManipulator>(std::boolalpha), true) == "1");
}

void test_doubles() {
  CHECK(stream_all(1.0) == "1.0000000000000000e+00");
  CHECK(stream_all(-2.5) == "-2.5000000000000000e+00");
  CHECK(stream_all(0.0) == "0.0000000000000000e+00");
  CHECK(stream_all(-0.0) == "-0.0000000000000000e+00");
  CHECK(stream_all(1.0e10) == "1.0000000000000000e+10");
  CHECK(stream_all(1.0e-10) == "1.0000000000000000e-10");
  CHECK(stream_all(1.0e100) == "1.0000000000000000e+100");
  CHECK(stream_all(0.5f) == "5.0000000000000000e-01");
  CHECK(stream_all(std::numeric_limits<double>::quiet_NaN()) == "nan");
  CHECK(stream_all(std::numeric_limits<double>::infinity()) == "inf");
  CHECK(stream_all(-std::numeric_limits<double>::infinity()) == "-inf");

  // Check that the printed values are within a few roundoff errors of the
  // values, from denormals to the largest double
  const auto check_value = [](const double value) {
    CAPTURE(value);
    const std::string formatted = stream_all(value);
    CAPTURE(formatted);
    // Same format as "%.16e"
    REQUIRE(formatted.size() >= 22);
    const size_t offset = value < 0.0 ? 1 : 0;
    CHECK(formatted[offset + 1] == '.');
    CHECK(formatted[offset + 18] == 'e');
    const double parsed = std::strtod(formatted.c_str(), nullptr);
    if (std::abs(value) < std::numeric_limits<double>::min()) {
      // Denormals have fewer significant bits
      CHECK(std::abs(parsed - value) <=
            std::numeric_limits<double>::denorm_min() * 2.0);
    } else {
      CHECK(std::abs(parsed - value) <= 1.0e-15 * std::abs(value));
    }
  };
  check_value(std::numeric_limits<double>::max());
  check_value(std::numeric_limits<double>::min());
  check_value(std::numeric_limits<double>::denorm_min());
  check_value(-std::numeric_limits<double>::denorm_min() * 12345.0);
  check_value(0.1);
  check_value(9.9999999999999999);
  check_value(9.999999999999999e22);
  check_value(1.0 - std::numeric_limits<double>::epsilon());
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> mantissa_distribution(-10.0, 10.0);
  std::uniform_int_distribution<int> exponent_distribution(-320, 307);
  for (size_t i = 0; i < 10000; ++i) {
    check_value(mantissa_distribution(generator) *
                std::pow(10.0, exponent_distribution(generator)));
  }
}

void test_truncation() {
  const std::string long_string(DeviceMessage::capacity + 10, 'a');
  const std::string formatted = stream_all(long_string.c_str(), "b", 1.0);
  CHECK(formatted.size() == DeviceMessage::capacity - 1);
  CHECK(formatted ==
        std::string(DeviceMessage::capacity - 4, 'a') + std::string("..."));
  const std::string fits(DeviceMessage::capacity - 1, 'a');
  CHECK(stream_all(fits.c_str()) == fits);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ErrorHandling.DeviceMessage", "[Unit][ErrorHandling]") {
  test_types();
  test_doubles();
  test_truncation();
}
