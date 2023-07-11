// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct TriviallyResizable {};

struct Resizable {
  size_t size = 0;
  bool resized = false;
};

struct NoSize {};

template <typename T, int Label>
struct Tag {
  using type = T;
};
}  // namespace

// [SetNumberOfGridPointsImpl]
template <>
struct SetNumberOfGridPointsImpls::SetNumberOfGridPointsImpl<
    TriviallyResizable> {
  static constexpr bool is_trivial = true;
};

template <>
struct SetNumberOfGridPointsImpls::SetNumberOfGridPointsImpl<Resizable> {
  static constexpr bool is_trivial = false;
  static void apply(const gsl::not_null<Resizable*> result, const size_t size) {
    result->size = size;
    result->resized = true;
  }
};
// [SetNumberOfGridPointsImpl]

namespace MakeWithValueImpls {
template <>
struct NumberOfPoints<Resizable> {
  static size_t apply(const Resizable& input) { return input.size; }
};
}  // namespace MakeWithValueImpls

SPECTRE_TEST_CASE("Unit.Utilities.SetNumberOfGridPoints", "[Utilities][Unit]") {
  Resizable pattern3;
  pattern3.size = 3;

  {
    TriviallyResizable x;
    set_number_of_grid_points(make_not_null(&x), 3_st);
    set_number_of_grid_points(make_not_null(&x), pattern3);
    set_number_of_grid_points(make_not_null(&x), TriviallyResizable{});
  }

  {
    Resizable r;
    set_number_of_grid_points(make_not_null(&r), 3_st);
    CHECK(r.size == 3);
    CHECK(r.resized);
  }
  {
    Resizable r;
    r.size = 3;
    set_number_of_grid_points(make_not_null(&r), 3_st);
    CHECK(r.size == 3);
    CHECK(not r.resized);
  }
  {
    Resizable r;
    set_number_of_grid_points(make_not_null(&r), pattern3);
    CHECK(r.size == 3);
    CHECK(r.resized);
  }
  {
    Resizable r;
    r.size = 3;
    set_number_of_grid_points(make_not_null(&r), pattern3);
    CHECK(r.size == 3);
    CHECK(not r.resized);
  }

  {
    double x = 2.0;
    set_number_of_grid_points(make_not_null(&x), 3_st);
    CHECK(x == 2.0);
  }
  {
    double x = 2.0;
    set_number_of_grid_points(make_not_null(&x), NoSize{});
    CHECK(x == 2.0);
  }

  {
    std::complex<double> x(2.0, 4.0);
    set_number_of_grid_points(make_not_null(&x), 3_st);
    CHECK(x == std::complex<double>(2.0, 4.0));
  }
  {
    std::complex<double> x(2.0, 4.0);
    set_number_of_grid_points(make_not_null(&x), NoSize{});
    CHECK(x == std::complex<double>(2.0, 4.0));
  }

  {
    std::array<Resizable, 3> x;
    x[0].size = 2;
    x[1].size = 2;
    x[2].size = 2;
    set_number_of_grid_points(make_not_null(&x), pattern3);
    CHECK(x[0].size == 3);
    CHECK(x[1].size == 3);
    CHECK(x[2].size == 3);
    CHECK(x[0].resized);
    CHECK(x[1].resized);
    CHECK(x[2].resized);
  }
  {
    std::array<Resizable, 3> x;
    x[0].size = 3;
    x[1].size = 3;
    x[2].size = 3;
    set_number_of_grid_points(make_not_null(&x), pattern3);
    CHECK(x[0].size == 3);
    CHECK(x[1].size == 3);
    CHECK(x[2].size == 3);
    CHECK(not x[0].resized);
    CHECK(not x[1].resized);
    CHECK(not x[2].resized);
  }
  {
    std::array<double, 3> x{1.0, 2.0, 3.0};
    set_number_of_grid_points(make_not_null(&x), NoSize{});
    CHECK(x == std::array{1.0, 2.0, 3.0});
  }
  {
    std::array<double, 3> x{1.0, 2.0, 3.0};
    set_number_of_grid_points(make_not_null(&x), 1_st);
    CHECK(x == std::array{1.0, 2.0, 3.0});
  }
  {
    std::array<Resizable, 0> x{};
    set_number_of_grid_points(make_not_null(&x), 1_st);
  }

  {
    std::vector<Resizable> x{{2, false}, {2, false}, {2, false}};
    set_number_of_grid_points(make_not_null(&x), pattern3);
    CHECK(x[0].size == 3);
    CHECK(x[1].size == 3);
    CHECK(x[2].size == 3);
    CHECK(x[0].resized);
    CHECK(x[1].resized);
    CHECK(x[2].resized);
  }
  {
    std::vector<Resizable> x{{3, false}, {3, false}, {3, false}};
    set_number_of_grid_points(make_not_null(&x), pattern3);
    CHECK(x[0].size == 3);
    CHECK(x[1].size == 3);
    CHECK(x[2].size == 3);
    CHECK(not x[0].resized);
    CHECK(not x[1].resized);
    CHECK(not x[2].resized);
  }
  {
    std::vector<double> x{1.0, 2.0, 3.0};
    set_number_of_grid_points(make_not_null(&x), NoSize{});
    CHECK(x == std::vector{1.0, 2.0, 3.0});
  }
  {
    std::vector<double> x{1.0, 2.0, 3.0};
    set_number_of_grid_points(make_not_null(&x), 1_st);
    CHECK(x == std::vector{1.0, 2.0, 3.0});
  }

  {
    tuples::TaggedTuple<Tag<Resizable, 0>, Tag<Resizable, 1>> x;
    set_number_of_grid_points(make_not_null(&x), pattern3);
    CHECK(tuples::get<Tag<Resizable, 0>>(x).size == 3);
    CHECK(tuples::get<Tag<Resizable, 1>>(x).size == 3);
    CHECK(tuples::get<Tag<Resizable, 0>>(x).resized);
    CHECK(tuples::get<Tag<Resizable, 1>>(x).resized);
  }
  {
    tuples::TaggedTuple<Tag<Resizable, 0>, Tag<Resizable, 1>> x;
    tuples::get<Tag<Resizable, 0>>(x).size = 3;
    tuples::get<Tag<Resizable, 1>>(x).size = 3;
    set_number_of_grid_points(make_not_null(&x), pattern3);
    CHECK(tuples::get<Tag<Resizable, 0>>(x).size == 3);
    CHECK(tuples::get<Tag<Resizable, 1>>(x).size == 3);
    CHECK(not tuples::get<Tag<Resizable, 0>>(x).resized);
    CHECK(not tuples::get<Tag<Resizable, 1>>(x).resized);
  }
  {
    tuples::TaggedTuple<Tag<Resizable, 0>, Tag<Resizable, 1>> x;
    set_number_of_grid_points(make_not_null(&x), 3_st);
    CHECK(tuples::get<Tag<Resizable, 0>>(x).size == 3);
    CHECK(tuples::get<Tag<Resizable, 1>>(x).size == 3);
    CHECK(tuples::get<Tag<Resizable, 0>>(x).resized);
    CHECK(tuples::get<Tag<Resizable, 1>>(x).resized);
  }
  {
    // We've checked resizing double and complex<double> above.  This
    // just checks that the correct overloads are called so these
    // compile.
    tuples::TaggedTuple<Tag<double, 0>, Tag<std::complex<double>, 0>> x;
    set_number_of_grid_points(make_not_null(&x), 3_st);
    set_number_of_grid_points(make_not_null(&x), pattern3);
    set_number_of_grid_points(make_not_null(&x), NoSize{});
  }
}
