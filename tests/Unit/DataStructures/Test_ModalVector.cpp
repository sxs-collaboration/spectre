// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.ModalVector",
                  "[DataStructures][Unit]") {
  ModalVector a{2};
  CHECK(a.size() == 2);
  ModalVector b{2, 10.0};
  CHECK(b.size() == 2);
  for (size_t i = 0; i < b.size(); ++i) {
    INFO(i);
    CHECK(b[i] == 10.0);
  }

  ModalVector t(5, 10.0);
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
  ModalVector t2{1.43, 2.83, 3.94, 7.85};
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
  ModalVector t_move_assignment = std::move(t_copy);
  CHECK(t_move_assignment.is_owning());
  ModalVector t_move_constructor = std::move(t_move_assignment);
  CHECK(t_move_constructor.is_owning());
}

SPECTRE_TEST_CASE("Unit.Serialization.ModalVector",
                  "[DataStructures][Unit][Serialization]") {
  const size_t npts = 10;
  ModalVector t(npts), tgood(npts);
  std::iota(t.begin(), t.end(), 1.2);
  std::iota(tgood.begin(), tgood.end(), 1.2);
  CHECK(tgood == t);
  CHECK(t.is_owning());
  CHECK(tgood.is_owning());
  const ModalVector serialized = serialize_and_deserialize(t);
  CHECK(tgood == t);
  CHECK(serialized == tgood);
  CHECK(serialized.is_owning());
  CHECK(serialized.data() != t.data());
  CHECK(t.is_owning());
}

SPECTRE_TEST_CASE("Unit.Serialization.ModalVector_Ref",
                  "[DataStructures][Unit][Serialization]") {
  const size_t npts = 10;
  ModalVector t(npts);
  std::iota(t.begin(), t.end(), 4.3);
  ModalVector t2;
  t2.set_data_ref(&t);
  CHECK(t.is_owning());
  CHECK_FALSE(t2.is_owning());
  CHECK(t2 == t);
  const ModalVector serialized = serialize_and_deserialize(t);
  CHECK(t2 == t);
  CHECK(serialized == t2);
  CHECK(serialized.is_owning());
  CHECK(serialized.data() != t.data());
  CHECK(t.is_owning());
  const ModalVector serialized2 = serialize_and_deserialize(t2);
  CHECK(t2 == t);
  CHECK(serialized2 == t2);
  CHECK(serialized2.is_owning());
  CHECK(serialized2.data() != t2.data());
  CHECK_FALSE(t2.is_owning());
}

SPECTRE_TEST_CASE("Unit.DataStructures.ModalVector_Ref",
                  "[DataStructures][Unit]") {
  ModalVector data{1.43, 2.83, 3.94, 7.85};
  ModalVector t;
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
  ModalVector t_copy;
  t_copy.set_data_ref(&t);
  test_move_semantics(std::move(t), t_copy);
  ModalVector t_move_assignment = std::move(t_copy);
  CHECK(not t_move_assignment.is_owning());
  ModalVector t_move_constructor = std::move(t_move_assignment);
  CHECK(not t_move_constructor.is_owning());
  {
    ModalVector data_2{1.43, 2.83, 3.94, 7.85};
    ModalVector data_2_ref;
    data_2_ref.set_data_ref(&data_2);
    ModalVector data_3{2.43, 3.83, 4.94, 8.85};
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
    ModalVector owned_data;
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
[[noreturn]] SPECTRE_TEST_CASE(
                "Unit.DataStructures.ModalVector.ref_diff_size",
                "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  ModalVector data{1.43, 2.83, 3.94, 7.85};
  ModalVector data_ref;
  data_ref.set_data_ref(&data);
  ModalVector data2{1.43, 2.83, 3.94};
  data_ref = data2;
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.ModalVector.move_ref_diff_size",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  ModalVector data{1.43, 2.83, 3.94, 7.85};
  ModalVector data_ref;
  data_ref.set_data_ref(&data);
  ModalVector data2{1.43, 2.83, 3.94};
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

SPECTRE_TEST_CASE("Unit.DataStructures.ModalVector.MathAfterMove",
                  "[Unit][DataStructures]") {
  ModalVector m0(10, 3.0), m1(10, 9.0);
  {
    ModalVector a(10, 2.0);
    ModalVector b{};
    b = std::move(a);
    b = m0 + m1;
    check_vectors(b, ModalVector(10, 12.0));
    // clang-tidy: use after move (intentional here)
    CHECK(a.size() == 0);  // NOLINT
    CHECK(a.is_owning());
    a = m0 * m1;
    check_vectors(a, ModalVector(10, 27.0));
    check_vectors(b, ModalVector(10, 12.0));
  }

  {
    ModalVector a(10, 2.0);
    ModalVector b{};
    b = std::move(a);
    a = m0 + m1;
    check_vectors(b, ModalVector(10, 2.0));
    check_vectors(a, ModalVector(10, 12.0));
  }

  {
    ModalVector a(10, 2.0);
    ModalVector b{std::move(a)};
    b = m0 + m1;
    CHECK(b.size() == 10);
    check_vectors(b, ModalVector(10, 12.0));
    // clang-tidy: use after move (intentional here)
    CHECK(a.size() == 0);  // NOLINT
    CHECK(a.is_owning());
    a = m0 * m1;
    check_vectors(a, ModalVector(10, 27.0));
    check_vectors(b, ModalVector(10, 12.0));
  }

  {
    ModalVector a(10, 2.0);
    ModalVector b{std::move(a)};
    a = m0 + m1;
    check_vectors(b, ModalVector(10, 2.0));
    check_vectors(a, ModalVector(10, 12.0));
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.ModalVector.Math",
                  "[Unit][DataStructures]") {
  const size_t num_pts = 19;
  ModalVector val{1., -2., 3., -4., 8., 12., -14.};
  ModalVector nine(num_pts, 9.0);
  ModalVector one(num_pts, 1.0);

  // Test unary minus
  check_vectors(-nine, ModalVector(num_pts, -9.0));

  check_vectors(nine + 2.0, ModalVector(num_pts, 11.0));
  check_vectors(2.0 + nine, ModalVector(num_pts, 11.0));
  check_vectors(nine - 2.0, ModalVector(num_pts, 7.0));
  check_vectors(2.0 - nine, ModalVector(num_pts, -7.0));
  check_vectors(nine + nine, ModalVector(num_pts, 18.0));
  check_vectors(nine + (one * nine), ModalVector(num_pts, 18.0));
  check_vectors((one * nine) + nine, ModalVector(num_pts, 18.0));
  check_vectors(nine - ModalVector(num_pts, 8.0), one);
  check_vectors(nine - (one * nine), ModalVector(num_pts, 0.0));
  check_vectors((one * nine) - nine, ModalVector(num_pts, 0.0));

  check_vectors(ModalVector(num_pts, 18.0), (one / 0.5) * nine);

  CHECK(-14 == min(val));
  CHECK(12 == max(val));
  check_vectors(ModalVector{1., 2., 3., 4., 8., 12., 14.}, abs(val));
  check_vectors(ModalVector{1., 2., 3., 4., 8., 12., 14.}, fabs(val));

  check_vectors(ModalVector(num_pts, 81.0), nine * nine);
  check_vectors(ModalVector(num_pts, 81.0), nine * (nine * one));
  check_vectors(ModalVector(num_pts, 81.0), (nine * nine) * one);
  check_vectors(ModalVector(num_pts, 81.0), 9.0 * nine);
  check_vectors(ModalVector(num_pts, 81.0), nine * 9.0);
  check_vectors(ModalVector(num_pts, 1.0), nine / 9.0);

  ModalVector dummy(nine * nine * 1.0);
  check_vectors(ModalVector(num_pts, 81.0), dummy);

  // Test assignment
  ModalVector test_81(num_pts, -1.0);
  test_81 = nine * nine;
  check_vectors(ModalVector(num_pts, 81.0), test_81);
  CHECK(test_81.is_owning());
  ModalVector test_81_ref(test_81.data(), test_81.size());
  test_81_ref = 0.0;
  test_81 = 0.0;
  test_81_ref = nine * nine;
  check_vectors(ModalVector(num_pts, 81.0), test_81);
  CHECK(test_81.is_owning());
  check_vectors(ModalVector(num_pts, 81.0), test_81_ref);
  CHECK_FALSE(test_81_ref.is_owning());
  ModalVector second_81(num_pts);
  second_81 = test_81;
  check_vectors(ModalVector(num_pts, 81.0), second_81);
  CHECK(second_81.is_owning());
  test_81_ref = 0.0;
  check_vectors(ModalVector(num_pts, 0.0), test_81_ref);
  test_81_ref = second_81;
  check_vectors(ModalVector(num_pts, 81.0), test_81_ref);
  second_81 = 0.0;
  test_81_ref.set_data_ref(&test_81);
  second_81 = std::move(test_81_ref);
  check_vectors(ModalVector(num_pts, 81.0), second_81);
  CHECK_FALSE(second_81.is_owning());
  second_81 = 0.0;
  check_vectors(ModalVector(num_pts, 0.0), second_81);
  test_81 = 81.0;
  check_vectors(ModalVector(num_pts, 81.0), second_81);
  CHECK_FALSE(second_81.is_owning());

  test_81 = 81.0;
  ModalVector test_081;
  test_081.set_data_ref(&test_81);
  check_vectors(ModalVector(num_pts, 81.0), test_081);
  CHECK_FALSE(test_081.is_owning());

  ModalVector test_assignment(num_pts, 7.0);
  test_assignment += nine;
  check_vectors(ModalVector(num_pts, 16.0), test_assignment);
  test_assignment += 3.0;
  check_vectors(ModalVector(num_pts, 19.0), test_assignment);
  test_assignment += (nine * nine);
  check_vectors(ModalVector(num_pts, 100.0), test_assignment);

  test_assignment = 7.0;
  test_assignment -= nine;
  check_vectors(ModalVector(num_pts, -2.0), test_assignment);
  test_assignment -= 3.0;
  check_vectors(ModalVector(num_pts, -5.0), test_assignment);
  test_assignment -= nine * nine;
  check_vectors(ModalVector(num_pts, -86.0), test_assignment);

  test_assignment = 2.0;
  test_assignment *= 3.0;
  check_vectors(ModalVector(num_pts, 6.0), test_assignment);
  test_assignment *= nine;
  check_vectors(ModalVector(num_pts, 54.0), test_assignment);
  test_assignment = 1.0;
  test_assignment *= nine * nine;
  check_vectors(ModalVector(num_pts, 81.0), test_assignment);

  test_assignment = 2.0;
  test_assignment /= 2.0;
  check_vectors(ModalVector(num_pts, 1.0), test_assignment);

  // Test assignment where the RHS is an expression that contains the LHS
  ModalVector x(num_pts, 4.);
  x *= x;
  check_vectors(ModalVector(num_pts, 16.0), x);

  // Test addition of arrays of ModalVectors to arrays of doubles.
  const ModalVector t1{0.0, 1.0, 2.0, 3.0};
  const ModalVector t2{-0.1, -0.2, -0.3, -0.4};
  const ModalVector t3{5.0, 4.0, 3.0, 2.0};
  const ModalVector e1{10.0, 11.0, 12.0, 13.0};
  const ModalVector e2{19.9, 19.8, 19.7, 19.6};
  const ModalVector e3{35.0, 34.0, 33.0, 32.0};
  const ModalVector e4{-10.0, -9.0, -8.0, -7.0};
  const ModalVector e5{-20.1, -20.2, -20.3, -20.4};
  const ModalVector e6{-25.0, -26.0, -27.0, -28.0};
  const ModalVector e7{10.0, 12.0, 14.0, 16.0};
  const ModalVector e8{19.8, 19.6, 19.4, 19.2};
  const ModalVector e9{40.0, 38.0, 36.0, 34.0};
  const ModalVector e10{-10.0, -10.0, -10.0, -10.0};
  const ModalVector e11{-20.0, -20.0, -20.0, -20.0};
  const ModalVector e12{-30.0, -30.0, -30.0, -30.0};

  const std::array<double, 3> point{{10.0, 20.0, 30.0}};
  std::array<ModalVector, 3> points{{t1, t2, t3}};
  const std::array<ModalVector, 3> expected{{e1, e2, e3}};
  const std::array<ModalVector, 3> expected2{{e4, e5, e6}};
  std::array<ModalVector, 3> dvectors1{{t1, t2, t3}};
  const std::array<ModalVector, 3> dvectors2{{e1, e2, e3}};
  const std::array<ModalVector, 3> expected3{{e7, e8, e9}};
  const std::array<ModalVector, 3> expected4{{e10, e11, e12}};
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

  // Test calculation of magnitude of ModalVector
  const std::array<ModalVector, 1> d1{{ModalVector{-2.5, 3.4}}};
  const ModalVector expected_d1{2.5, 3.4};
  const auto magnitude_d1 = magnitude(d1);
  CHECK_ITERABLE_APPROX(expected_d1, magnitude_d1);
  const std::array<ModalVector, 2> d2{{ModalVector(2, 3.),
                                             ModalVector(2, 4.)}};
  const ModalVector expected_d2(2, 5.);
  const auto magnitude_d2 = magnitude(d2);
  CHECK_ITERABLE_APPROX(expected_d2, magnitude_d2);
  const std::array<ModalVector, 3> d3{
      {ModalVector(2, 3.), ModalVector(2, -4.),
       ModalVector(2, 12.)}};
  const ModalVector expected_d3(2, 13.);
  const auto magnitude_d3 = magnitude(d3);
  CHECK_ITERABLE_APPROX(expected_d3, magnitude_d3);
}

SPECTRE_TEST_CASE("Unit.DataStructures.ModalVector.MathWithDataVector",
                  "[Unit][DataStructures]") {
  const size_t num_pts = 23;
  ModalVector nine(num_pts, 9.0);
  CHECK(nine.is_owning());
  DataVector eight(num_pts, 8.0);

  // Test math with DataVector
  ModalVector test_72(num_pts, -1.0);
  test_72 = nine * eight;
  check_vectors(ModalVector(num_pts, 72.0), test_72);
  CHECK(test_72.is_owning());
  test_72 = eight * nine;
  check_vectors(ModalVector(num_pts, 72.0), test_72);
  CHECK(test_72.is_owning());
  test_72 = test_72 / eight;
  check_vectors(ModalVector(num_pts, 9.0), test_72);
  test_72 = nine;
  test_72 *= eight;
  check_vectors(ModalVector(num_pts, 72.0), test_72);
  CHECK(test_72.is_owning());
  test_72 /= eight;
  check_vectors(ModalVector(num_pts, 9.0), test_72);
  CHECK(test_72.is_owning());
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.ModalVector.ExpressionAssignError",
    "[Unit][DataStructures]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  ModalVector one(10, 1.0);
  ModalVector one_ref(one.data(), one.size());
  ModalVector one_b(2, 1.0);
  one_ref = (one_b * one_b);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
