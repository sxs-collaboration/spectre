// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.DataVector", "[DataStructures][Unit]") {
  DataVector a{2};
  CHECK(a.size() == 2);
  DataVector b{2, 10.0};
  CHECK(b.size() == 2);
  for (size_t i = 0; i < b.size(); ++i) {
    INFO(i);
    CHECK(b[i] == 10.0);
  }

  DataVector t(5, 10.0);
  CHECK(t.size() == 5);
  for (size_t i = 0; i < t.size(); ++i) {
    INFO(i);
    CHECK(t[i] == 10.0);
  }
  for (const auto& p : t) {
    CHECK(p == 10.0);
  }
  for (auto& p : t) {
    CHECK(p == 10.0);
  }
  DataVector t2{1.43, 2.83, 3.94, 7.85};
  CHECK(t2.size() == 4);
  CHECK(t2.is_owning());
  CHECK(t2[0] == 1.43);
  CHECK(t2[1] == 2.83);
  CHECK(t2[2] == 3.94);
  CHECK(t2[3] == 7.85);
  test_copy_semantics(t);
  auto t_copy = t;
  CHECK(t_copy.is_owning());
  test_move_semantics(std::move(t), t_copy);
  DataVector t_move_assignment = std::move(t_copy);
  CHECK(t_move_assignment.is_owning());
  DataVector t_move_constructor = std::move(t_move_assignment);
  CHECK(t_move_constructor.is_owning());
}

SPECTRE_TEST_CASE("Unit.Serialization.DataVector",
                  "[DataStructures][Unit][Serialization]") {
  const size_t npts = 10;
  DataVector t(npts), tgood(npts);
  std::iota(t.begin(), t.end(), 1.2);
  std::iota(tgood.begin(), tgood.end(), 1.2);
  CHECK(tgood == t);
  CHECK(t.is_owning());
  CHECK(tgood.is_owning());
  const DataVector serialized = serialize_and_deserialize(t);
  CHECK(tgood == t);
  CHECK(serialized == tgood);
  CHECK(serialized.is_owning());
  CHECK(serialized.data() != t.data());
  CHECK(t.is_owning());
}

SPECTRE_TEST_CASE("Unit.Serialization.DataVector_Ref",
                  "[DataStructures][Unit][Serialization]") {
  const size_t npts = 10;
  DataVector t(npts);
  std::iota(t.begin(), t.end(), 4.3);
  DataVector t2;
  t2.set_data_ref(&t);
  CHECK(t.is_owning());
  CHECK_FALSE(t2.is_owning());
  CHECK(t2 == t);
  const DataVector serialized = serialize_and_deserialize(t);
  CHECK(t2 == t);
  CHECK(serialized == t2);
  CHECK(serialized.is_owning());
  CHECK(serialized.data() != t.data());
  CHECK(t.is_owning());
  const DataVector serialized2 = serialize_and_deserialize(t2);
  CHECK(t2 == t);
  CHECK(serialized2 == t2);
  CHECK(serialized2.is_owning());
  CHECK(serialized2.data() != t2.data());
  CHECK_FALSE(t2.is_owning());
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataVector_Ref",
                  "[DataStructures][Unit]") {
  DataVector data{1.43, 2.83, 3.94, 7.85};
  DataVector t;
  t.set_data_ref(&data);
  CHECK(not t.is_owning());
  CHECK(data.is_owning());
  CHECK(t.data() == data.data());
  CHECK(t.size() == 4);
  CHECK(t[0] == 1.43);
  CHECK(t[1] == 2.83);
  CHECK(t[2] == 3.94);
  CHECK(t[3] == 7.85);
  test_copy_semantics(t);
  DataVector t_copy;
  t_copy.set_data_ref(&t);
  test_move_semantics(std::move(t), t_copy);
  DataVector t_move_assignment = std::move(t_copy);
  CHECK(not t_move_assignment.is_owning());
  DataVector t_move_constructor = std::move(t_move_assignment);
  CHECK(not t_move_constructor.is_owning());
  {
    DataVector data_2{1.43, 2.83, 3.94, 7.85};
    DataVector data_2_ref;
    data_2_ref.set_data_ref(&data_2);
    DataVector data_3{2.43, 3.83, 4.94, 8.85};
    data_2_ref = std::move(data_3);
    CHECK(data_2[0] == 2.43);
    CHECK(data_2[1] == 3.83);
    CHECK(data_2[2] == 4.94);
    CHECK(data_2[3] == 8.85);
// Intentionally testing self-move
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif  // defined(__clang__)
    data_2_ref = std::move(data_2_ref);
#ifdef __clang__
#pragma GCC diagnostic pop
#endif  // defined(__clang__)
    CHECK(data_2[0] == 2.43);
    CHECK(data_2[1] == 3.83);
    CHECK(data_2[2] == 4.94);
    CHECK(data_2[3] == 8.85);
    DataVector owned_data;
    // clang-tidy: false positive, used after it was moved
    owned_data = data_2_ref;  // NOLINT
    CHECK(owned_data[0] == 2.43);
    CHECK(owned_data[1] == 3.83);
    CHECK(owned_data[2] == 4.94);
    CHECK(owned_data[3] == 8.85);
    // Test operator!=
    CHECK_FALSE(owned_data != data_2_ref);
  }
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.DataStructures.DataVector.ref_diff_size",
                               "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DataVector data{1.43, 2.83, 3.94, 7.85};
  DataVector data_ref;
  data_ref.set_data_ref(&data);
  DataVector data2{1.43, 2.83, 3.94};
  data_ref = data2;
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.DataVector.move_ref_diff_size",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DataVector data{1.43, 2.83, 3.94, 7.85};
  DataVector data_ref;
  data_ref.set_data_ref(&data);
  DataVector data2{1.43, 2.83, 3.94};
  data_ref = std::move(data2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

namespace {
template <typename T1, typename T2>
void check_vectors(const T1& t1, const T2& t2) {
  CHECK_ITERABLE_APPROX(t1, t2);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataVector.MathAfterMove",
                  "[Unit][DataStructures]") {
  DataVector m0(10, 3.0), m1(10, 9.0);
  {
    DataVector a(10, 2.0);
    DataVector b{};
    b = std::move(a);
    b = m0 + m1;
    check_vectors(b, DataVector(10, 12.0));
    // clang-tidy: use after move (intentional here)
    CHECK(a.size() == 0);  // NOLINT
    CHECK(a.is_owning());
    a = m0 * m1;
    check_vectors(a, DataVector(10, 27.0));
    check_vectors(b, DataVector(10, 12.0));
  }

  {
    DataVector a(10, 2.0);
    DataVector b{};
    b = std::move(a);
    a = m0 + m1;
    check_vectors(b, DataVector(10, 2.0));
    check_vectors(a, DataVector(10, 12.0));
  }

  {
    DataVector a(10, 2.0);
    DataVector b{std::move(a)};
    b = m0 + m1;
    CHECK(b.size() == 10);
    check_vectors(b, DataVector(10, 12.0));
    // clang-tidy: use after move (intentional here)
    CHECK(a.size() == 0);  // NOLINT
    CHECK(a.is_owning());
    a = m0 * m1;
    check_vectors(a, DataVector(10, 27.0));
    check_vectors(b, DataVector(10, 12.0));
  }

  {
    DataVector a(10, 2.0);
    DataVector b{std::move(a)};
    a = m0 + m1;
    check_vectors(b, DataVector(10, 2.0));
    check_vectors(a, DataVector(10, 12.0));
  }
}

namespace {
enum class UseRefWrap { None, Cref, Ref };

template <UseRefWrap Wrap, class T,
          Requires<Wrap == UseRefWrap::Cref> = nullptr>
decltype(auto) wrap(const T& t) noexcept {
  return std::cref(t);
}
template <UseRefWrap Wrap, class T, Requires<Wrap == UseRefWrap::Ref> = nullptr>
decltype(auto) wrap(T& t) noexcept {
  return std::ref(t);
}
template <UseRefWrap Wrap, class T,
          Requires<Wrap == UseRefWrap::None> = nullptr>
decltype(auto) wrap(const T& t) noexcept {
  return t;
}

// Wrap is used to wrap values in a std::reference_wrapper using std::cref and
// std::ref, or to not wrap at all. This is done to verify that all math
// operations work transparently with a `std::reference_wrapper` too.
template <UseRefWrap WrapLeftOp, UseRefWrap WrapRightOp>
void test_datavector_math() noexcept {
  constexpr size_t num_pts = 19;
  DataVector val{1., 2., 3., -4., 8., 12., -14.};
  DataVector one(num_pts, 1.0);
  DataVector eight(num_pts, 8.0);
  DataVector nine(num_pts, 9.0);

  // Test unary minus
  check_vectors(-wrap<WrapLeftOp>(nine), DataVector(num_pts, -9.0));

  check_vectors(wrap<WrapLeftOp>(nine) + 2.0, DataVector(num_pts, 11.0));
  check_vectors(2.0 + wrap<WrapLeftOp>(nine), DataVector(num_pts, 11.0));
  check_vectors(wrap<WrapLeftOp>(nine) - 2.0, DataVector(num_pts, 7.0));
  check_vectors(2.0 - wrap<WrapLeftOp>(nine), DataVector(num_pts, -7.0));
  check_vectors(wrap<WrapLeftOp>(nine) + wrap<WrapRightOp>(nine),
                DataVector(num_pts, 18.0));
  check_vectors(wrap<WrapLeftOp>(nine) +
                    (wrap<WrapLeftOp>(one) * wrap<WrapRightOp>(nine)),
                DataVector(num_pts, 18.0));
  check_vectors((wrap<WrapLeftOp>(one) * wrap<WrapLeftOp>(nine)) +
                    wrap<WrapRightOp>(nine),
                DataVector(num_pts, 18.0));
  check_vectors(wrap<WrapLeftOp>(nine) - DataVector(num_pts, 8.0),
                DataVector(num_pts, 1.0));
  check_vectors(wrap<WrapLeftOp>(nine) -
                    (wrap<WrapRightOp>(one) * wrap<WrapLeftOp>(nine)),
                DataVector(num_pts, 0.0));
  check_vectors((wrap<WrapLeftOp>(one) * wrap<WrapLeftOp>(nine)) -
                    wrap<WrapRightOp>(nine),
                DataVector(num_pts, 0.0));

  check_vectors(DataVector(num_pts, -1.0 / 9.0),
                -one / wrap<WrapRightOp>(nine));
  check_vectors(DataVector(num_pts, -8.0 / 9.0),
                -(wrap<WrapLeftOp>(nine) - wrap<WrapRightOp>(one)) /
                    wrap<WrapLeftOp>(nine));
  check_vectors(DataVector(num_pts, 18.0),
                (wrap<WrapLeftOp>(one) / 0.5) * wrap<WrapRightOp>(nine));
  check_vectors(DataVector(num_pts, 1.0), 9.0 / wrap<WrapRightOp>(nine));
  check_vectors(DataVector(num_pts, 1.0),
                (wrap<WrapLeftOp>(one) * 9.0) / wrap<WrapRightOp>(nine));

  check_vectors(DataVector(num_pts, 81.0),
                wrap<WrapLeftOp>(nine) * wrap<WrapRightOp>(nine));
  check_vectors(DataVector(num_pts, 81.0),
                wrap<WrapLeftOp>(nine) *
                    (wrap<WrapLeftOp>(nine) * wrap<WrapRightOp>(one)));
  check_vectors(DataVector(num_pts, 81.0),
                (wrap<WrapLeftOp>(nine) * wrap<WrapRightOp>(nine)) *
                    wrap<WrapLeftOp>(one));
  check_vectors(DataVector(num_pts, 81.0), 9.0 * wrap<WrapLeftOp>(nine));
  check_vectors(DataVector(num_pts, 81.0), wrap<WrapLeftOp>(nine) * 9.0);
  check_vectors(DataVector(num_pts, 1.0), wrap<WrapLeftOp>(nine) / 9.0);
  check_vectors(DataVector(num_pts, 1.0),
                wrap<WrapLeftOp>(nine) / (wrap<WrapRightOp>(one) * 9.0));
  check_vectors(DataVector(num_pts, 9.0),
                (wrap<WrapLeftOp>(one) * wrap<WrapLeftOp>(nine)) /
                    wrap<WrapRightOp>(one));

  CHECK(-14 == min(wrap<WrapLeftOp>(val)));
  CHECK(12 == max(wrap<WrapLeftOp>(val)));
  check_vectors(DataVector{1., 2., 3., 4., 8., 12., 14.},
                abs(wrap<WrapLeftOp>(val)));
  check_vectors(DataVector{1., 2., 3., 4., 8., 12., 14.},
                fabs(wrap<WrapLeftOp>(val)));

  const DataVector step_function_data{-12.3, 2.0, -4.0, 0.0, 7.0, -8.0};
  check_vectors(step_function(wrap<WrapLeftOp>(step_function_data)),
                DataVector{0.0, 1.0, 0.0, 1.0, 1.0, 0.0});

  check_vectors(sqrt(wrap<WrapLeftOp>(nine)), DataVector(num_pts, 3.0));
  check_vectors(invsqrt(wrap<WrapLeftOp>(nine)),
                DataVector(num_pts, 1.0 / 3.0));
  check_vectors(cbrt(wrap<WrapLeftOp>(eight)), DataVector(num_pts, 2.0));
  check_vectors(invcbrt(wrap<WrapLeftOp>(eight)), DataVector(num_pts, 0.5));
  check_vectors(DataVector(num_pts, 81.0), pow(wrap<WrapLeftOp>(nine), 2));

  DataVector dummy(wrap<WrapLeftOp>(nine) * wrap<WrapLeftOp>(nine) * 1.0);
  check_vectors(DataVector(num_pts, 81.0), dummy);
  check_vectors(DataVector(num_pts, 81.0), pow<2>(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, 81.0),
                pow<2>(wrap<WrapLeftOp>(nine) * wrap<WrapRightOp>(one)));
  check_vectors(DataVector(num_pts, 1.0 / 81.0),
                DataVector(num_pts, 1.0) / pow<2>(wrap<WrapLeftOp>(nine)));

  check_vectors(DataVector(num_pts, exp(9.0)), exp(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, exp2(9.0)), exp2(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, pow<9>(10.0)),
                DataVector(exp10(wrap<WrapLeftOp>(nine))));
  check_vectors(DataVector(num_pts, log(9.0)), log(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, log2(9.0)), log2(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, log10(9.0)), log10(wrap<WrapLeftOp>(nine)));

  DataVector point_nine(num_pts, 0.9);
  check_vectors(DataVector(num_pts, hypot(0.9, 9.0)),
                hypot(wrap<WrapLeftOp>(point_nine), wrap<WrapRightOp>(nine)));
  check_vectors(DataVector(num_pts, hypot(0.9, 9.0)),
                hypot(wrap<WrapLeftOp>(point_nine),
                      wrap<WrapLeftOp>(one) * wrap<WrapRightOp>(nine)));
  check_vectors(DataVector(num_pts, hypot(0.9, 9.0)),
                hypot(wrap<WrapLeftOp>(one) * wrap<WrapRightOp>(point_nine),
                      wrap<WrapRightOp>(nine)));
  check_vectors(DataVector(num_pts, hypot(0.9, 9.0)),
                hypot(wrap<WrapLeftOp>(one) * wrap<WrapRightOp>(point_nine),
                      wrap<WrapLeftOp>(one) * wrap<WrapRightOp>(nine)));

  check_vectors(DataVector(num_pts, sin(9.0)), sin(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, cos(9.0)), cos(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, tan(9.0)), tan(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, asin(0.9)),
                asin(wrap<WrapLeftOp>(point_nine)));
  check_vectors(DataVector(num_pts, acos(0.9)),
                acos(wrap<WrapLeftOp>(point_nine)));
  check_vectors(DataVector(num_pts, atan(0.9)),
                atan(wrap<WrapLeftOp>(point_nine)));
  check_vectors(DataVector(num_pts, atan2(0.9, 9.0)),
                atan2(wrap<WrapLeftOp>(point_nine), wrap<WrapRightOp>(nine)));
  check_vectors(DataVector(num_pts, atan2(0.9, 9.0)),
                atan2(wrap<WrapLeftOp>(point_nine) * wrap<WrapRightOp>(one),
                      wrap<WrapRightOp>(nine)));
  check_vectors(DataVector(num_pts, atan2(0.9, 9.0)),
                atan2(wrap<WrapLeftOp>(point_nine),
                      wrap<WrapLeftOp>(nine) * wrap<WrapRightOp>(one)));
  check_vectors(DataVector(num_pts, atan2(0.9, 9.0)),
                atan2(wrap<WrapLeftOp>(point_nine) * wrap<WrapRightOp>(one),
                      wrap<WrapLeftOp>(nine) * wrap<WrapRightOp>(one)));

  check_vectors(DataVector(num_pts, sinh(9.0)), sinh(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, cosh(9.0)), cosh(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, tanh(9.0)), tanh(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, asinh(9.0)), asinh(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, acosh(9.0)), acosh(wrap<WrapLeftOp>(nine)));
  check_vectors(DataVector(num_pts, atanh(0.9)),
                atanh(wrap<WrapLeftOp>(point_nine)));

  check_vectors(DataVector(num_pts, erf(0.9)),
                erf(wrap<WrapLeftOp>(point_nine)));
  check_vectors(DataVector(num_pts, erfc(0.9)),
                erfc(wrap<WrapLeftOp>(point_nine)));

  // Test assignment
  DataVector test_81(num_pts, -1.0);
  // cannot assign to a reference_wrapper using an expr :(
  test_81 = wrap<WrapLeftOp>(nine) * wrap<WrapRightOp>(nine);
  check_vectors(DataVector(num_pts, 81.0), test_81);
  CHECK(test_81.is_owning());
  DataVector test_81_ref(test_81.data(), test_81.size());
  test_81_ref = 0.0;
  test_81 = 0.0;
  test_81_ref = wrap<WrapLeftOp>(nine) * wrap<WrapRightOp>(nine);
  check_vectors(DataVector(num_pts, 81.0), test_81);
  CHECK(test_81.is_owning());
  check_vectors(DataVector(num_pts, 81.0), test_81_ref);
  CHECK_FALSE(test_81_ref.is_owning());
  DataVector second_81(num_pts);
  second_81 = wrap<WrapLeftOp>(test_81);
  check_vectors(DataVector(num_pts, 81.0), second_81);
  CHECK(second_81.is_owning());
  test_81_ref = 0.0;
  check_vectors(DataVector(num_pts, 0.0), test_81_ref);
  test_81_ref = wrap<WrapLeftOp>(second_81);
  check_vectors(DataVector(num_pts, 81.0), test_81_ref);
  second_81 = 0.0;
  test_81_ref.set_data_ref(&test_81);
  second_81 = std::move(test_81_ref);
  check_vectors(DataVector(num_pts, 81.0), second_81);
  CHECK_FALSE(second_81.is_owning());
  second_81 = 0.0;
  check_vectors(DataVector(num_pts, 0.0), second_81);
  test_81 = 81.0;
  check_vectors(DataVector(num_pts, 81.0), second_81);
  CHECK_FALSE(second_81.is_owning());

  test_81 = 81.0;
  DataVector test_081;
  test_081.set_data_ref(&test_81);
  check_vectors(DataVector(num_pts, 81.0), test_081);
  CHECK_FALSE(test_081.is_owning());
  test_081 = square(point_nine);
  check_vectors(DataVector(num_pts, 0.81), test_081);
  CHECK_FALSE(test_081.is_owning());

  DataVector test_assignment(num_pts, 7.0);
  test_assignment += wrap<WrapLeftOp>(nine);
  check_vectors(DataVector(num_pts, 16.0), test_assignment);
  test_assignment += 3.0;
  check_vectors(DataVector(num_pts, 19.0), test_assignment);
  test_assignment += (wrap<WrapLeftOp>(nine) * wrap<WrapLeftOp>(nine));
  check_vectors(DataVector(num_pts, 100.0), test_assignment);

  test_assignment = 7.0;
  test_assignment -= wrap<WrapLeftOp>(nine);
  check_vectors(DataVector(num_pts, -2.0), test_assignment);
  test_assignment -= 3.0;
  check_vectors(DataVector(num_pts, -5.0), test_assignment);
  test_assignment -= wrap<WrapLeftOp>(nine) * wrap<WrapLeftOp>(nine);
  check_vectors(DataVector(num_pts, -86.0), test_assignment);

  test_assignment = 2.0;
  test_assignment *= 3.0;
  check_vectors(DataVector(num_pts, 6.0), test_assignment);
  test_assignment *= wrap<WrapLeftOp>(nine);
  check_vectors(DataVector(num_pts, 54.0), test_assignment);
  test_assignment = 1.0;
  test_assignment *= wrap<WrapLeftOp>(nine) * wrap<WrapLeftOp>(nine);
  check_vectors(DataVector(num_pts, 81.0), test_assignment);

  test_assignment = 2.0;
  test_assignment /= 2.0;
  check_vectors(DataVector(num_pts, 1.0), test_assignment);
  test_assignment /= DataVector(num_pts, 0.5);
  check_vectors(DataVector(num_pts, 2.0), test_assignment);
  test_assignment /= (DataVector(num_pts, 2.0) * DataVector(num_pts, 3.0));
  check_vectors(DataVector(num_pts, 1.0 / 3.0), test_assignment);

  // Test assignment where the RHS is an expression that contains the LHS
  DataVector x(num_pts, 4.);
  x += sqrt(wrap<WrapLeftOp>(x));
  check_vectors(DataVector(num_pts, 6.0), x);
  x -= sqrt(wrap<WrapLeftOp>(x) - 2.0);
  check_vectors(DataVector(num_pts, 4.0), x);
  x = sqrt(wrap<WrapLeftOp>(x));
  check_vectors(DataVector(num_pts, 2.0), x);
  x *= wrap<WrapLeftOp>(x);
  check_vectors(DataVector(num_pts, 4.0), x);
  x /= wrap<WrapLeftOp>(x);
  check_vectors(DataVector(num_pts, 1.0), x);

  // Test composition of constant expressions with DataVector math member
  // functions
  x = DataVector(num_pts, 2.);
  check_vectors(DataVector(num_pts, 0.82682181043180603),
                square(sin(wrap<WrapLeftOp>(x))));
  check_vectors(DataVector(num_pts, -0.072067555747765299),
                cube(cos(wrap<WrapLeftOp>(x))));
}

void test_datavector_array_math() noexcept {
  // Test addition of arrays of DataVectors to arrays of doubles.
  const DataVector t1{0.0, 1.0, 2.0, 3.0};
  const DataVector t2{-0.1, -0.2, -0.3, -0.4};
  const DataVector t3{5.0, 4.0, 3.0, 2.0};
  const DataVector e1{10.0, 11.0, 12.0, 13.0};
  const DataVector e2{19.9, 19.8, 19.7, 19.6};
  const DataVector e3{35.0, 34.0, 33.0, 32.0};
  const DataVector e4{-10.0, -9.0, -8.0, -7.0};
  const DataVector e5{-20.1, -20.2, -20.3, -20.4};
  const DataVector e6{-25.0, -26.0, -27.0, -28.0};
  const DataVector e7{10.0, 12.0, 14.0, 16.0};
  const DataVector e8{19.8, 19.6, 19.4, 19.2};
  const DataVector e9{40.0, 38.0, 36.0, 34.0};
  const DataVector e10{-10.0, -10.0, -10.0, -10.0};
  const DataVector e11{-20.0, -20.0, -20.0, -20.0};
  const DataVector e12{-30.0, -30.0, -30.0, -30.0};

  const std::array<double, 3> point{{10.0, 20.0, 30.0}};
  std::array<DataVector, 3> points{{t1, t2, t3}};
  const std::array<DataVector, 3> expected{{e1, e2, e3}};
  const std::array<DataVector, 3> expected2{{e4, e5, e6}};
  std::array<DataVector, 3> dvectors1{{t1, t2, t3}};
  const std::array<DataVector, 3> dvectors2{{e1, e2, e3}};
  const std::array<DataVector, 3> expected3{{e7, e8, e9}};
  const std::array<DataVector, 3> expected4{{e10, e11, e12}};
  CHECK(points + point == expected);
  CHECK(point + points == expected);
  CHECK(points - point == expected2);
  CHECK(point - points == -expected2);

  points += point;
  CHECK(points == expected);
  points -= point;
  points -= point;
  CHECK(points == expected2);

  CHECK_ITERABLE_APPROX(dvectors1 + dvectors2, expected3);
  CHECK_ITERABLE_APPROX(dvectors1 - dvectors2, expected4);

  dvectors1 += dvectors2;
  CHECK_ITERABLE_APPROX(dvectors1, expected3);
  dvectors1 -= dvectors2;
  dvectors1 -= dvectors2;
  CHECK_ITERABLE_APPROX(dvectors1, expected4);

  // Test calculation of magnitude of DataVector
  const std::array<DataVector, 1> d1{{DataVector{-2.5, 3.4}}};
  const DataVector expected_d1{2.5, 3.4};
  const auto magnitude_d1 = magnitude(d1);
  CHECK_ITERABLE_APPROX(expected_d1, magnitude_d1);
  const std::array<DataVector, 2> d2{{DataVector(2, 3.), DataVector(2, 4.)}};
  const DataVector expected_d2(2, 5.);
  const auto magnitude_d2 = magnitude(d2);
  CHECK_ITERABLE_APPROX(expected_d2, magnitude_d2);
  const std::array<DataVector, 3> d3{
      {DataVector(2, 3.), DataVector(2, -4.), DataVector(2, 12.)}};
  const DataVector expected_d3(2, 13.);
  const auto magnitude_d3 = magnitude(d3);
  CHECK_ITERABLE_APPROX(expected_d3, magnitude_d3);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataVector.Math",
                  "[Unit][DataStructures]") {
  test_datavector_math<UseRefWrap::Cref, UseRefWrap::Cref>();
  test_datavector_math<UseRefWrap::Cref, UseRefWrap::Ref>();
  test_datavector_math<UseRefWrap::Cref, UseRefWrap::None>();
  test_datavector_math<UseRefWrap::Ref, UseRefWrap::Cref>();
  test_datavector_math<UseRefWrap::Ref, UseRefWrap::Ref>();
  test_datavector_math<UseRefWrap::Ref, UseRefWrap::None>();
  test_datavector_math<UseRefWrap::None, UseRefWrap::Cref>();
  test_datavector_math<UseRefWrap::None, UseRefWrap::Ref>();
  test_datavector_math<UseRefWrap::None, UseRefWrap::None>();

  test_datavector_array_math();
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.DataVector.ExpressionAssignError",
    "[Unit][DataStructures]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DataVector one(10, 1.0);
  DataVector one_ref(one.data(), one.size());
  DataVector one_b(2, 1.0);
  one_ref = (one_b * one_b);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
