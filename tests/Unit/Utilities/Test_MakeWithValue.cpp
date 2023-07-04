// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>

#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Makeable {
  size_t size = 0;
  double value = 0.0;
};

bool operator==(const Makeable& a, const Makeable& b) {
  return a.size == b.size and a.value == b.value;
}
}  // namespace

namespace MakeWithValueImpls {
template <>
struct MakeWithSize<Makeable> {
  static Makeable apply(const size_t size, const double value) {
    return Makeable{size, value};
  }
};

template <>
struct NumberOfPoints<Makeable> {
  static size_t apply(const Makeable& input) { return input.size; }
};
}  // namespace MakeWithValueImpls

namespace {
namespace Tags {
struct Makeable {
  using type = ::Makeable;
};

struct Makeable2 {
  using type = ::Makeable;
};

struct Double {
  using type = double;
};
}  // namespace Tags

template <typename R, typename T, typename ValueType>
void check_make_with_value(const R& expected, const T& input,
                           const ValueType value) {
  const auto computed = make_with_value<R>(input, value);
  CHECK(expected == computed);
}

void test_make_tagged_tuple() {
  check_make_with_value(tuples::TaggedTuple<Tags::Double>(-5.7), 0.0, -5.7);
  for (size_t n_pts = 1; n_pts < 4; ++n_pts) {
    check_make_with_value(
        tuples::TaggedTuple<Tags::Makeable>(Makeable{n_pts, -5.7}),
        Makeable{n_pts, 0.0}, -5.7);

    check_make_with_value(
        tuples::TaggedTuple<Tags::Makeable, Tags::Makeable2, Tags::Double>(
            Makeable{n_pts, 3.8}, Makeable{n_pts, 3.8}, 3.8),
        Makeable{n_pts, 0.0}, 3.8);
    check_make_with_value(Makeable{n_pts, 3.8},
                          tuples::TaggedTuple<Tags::Makeable, Tags::Makeable2>(
                              Makeable{n_pts, 0.0}, Makeable{n_pts, 1.0}),
                          3.8);
  }

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      make_with_value<Makeable>(
          tuples::TaggedTuple<Tags::Makeable, Tags::Makeable2>(
              Makeable{1, 0.0}, Makeable{2, 0.0}),
          0.0),
      Catch::Contains("Inconsistent number of points in tuple entries"));
#endif  // SPECTRE_DEBUG
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.MakeWithValue",
                  "[DataStructures][Unit]") {
  check_make_with_value(Makeable{2, 1.2}, 2_st, 1.2);
  check_make_with_value(Makeable{2, 1.2}, Makeable{2, 3.4}, 1.2);

  check_make_with_value(8.3, 1.3, 8.3);
  check_make_with_value(std::complex<double>(8.3, 2.5), 1.3,
                        std::complex<double>(8.3, 2.5));
  check_make_with_value(8.3, Makeable{8, 2.3}, 8.3);
  check_make_with_value(std::complex<double>(8.3, 2.5), Makeable{8, 2.3},
                        std::complex<double>(8.3, 2.5));

  check_make_with_value(make_array<4>(8.3), 1.3, 8.3);
  check_make_with_value(make_array<3>(Makeable{5, 8.3}), Makeable{5, 4.5}, 8.3);
  check_make_with_value(1.3, make_array<4>(8.3), 1.3);
  check_make_with_value(make_array<3>(Makeable{5, 8.3}),
                        make_array<4>(Makeable{5, 4.5}), 8.3);

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      make_with_value<Makeable>(make_array(Makeable{1, 0.0}, Makeable{2, 0.0}),
                                0.0),
      Catch::Contains("Inconsistent number of points in array entries"));
#endif  // SPECTRE_DEBUG

  test_make_tagged_tuple();
}
