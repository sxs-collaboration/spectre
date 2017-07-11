// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "tests/Unit/TestHelpers.hpp"

TEST_CASE("Unit.DataStructures.DataVector", "[DataStructures][Unit]") {
  DataVector t(5, 10.0);
  CHECK(t.size() == 5);
  // Not range-based so we can see which index is wrong if the test fails
  for (size_t i = 0; i < t.size(); ++i) {  // NOLINT
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
  test_move_semantics(t, t_copy);
  DataVector t_move_assignment = std::move(t_copy);
  CHECK(t_move_assignment.is_owning());
  DataVector t_move_constructor = std::move(t_move_assignment);
  CHECK(t_move_constructor.is_owning());
}

TEST_CASE("Unit.Serialization.DataVector",
          "[DataStructures][Unit][Serialization]") {
  const size_t npts = 10;
  DataVector t(npts), tgood(npts);
  std::iota(t.begin(), t.end(), 1.2);
  std::iota(tgood.begin(), tgood.end(), 1.2);
  CHECK(tgood == serialize_and_deserialize(t));
}

TEST_CASE("Unit.Serialization.DataVector_Ref",
          "[DataStructures][Unit][Serialization]") {
  const size_t npts = 10;
  DataVector t(npts);
  std::iota(t.begin(), t.end(), 4.3);
  DataVector t2;
  t2.set_data_ref(t);
  CHECK(t == serialize_and_deserialize(std::move(t2)));
}

TEST_CASE("Unit.DataStructures.DataVector_Ref", "[DataStructures][Unit]") {
  DataVector data{1.43, 2.83, 3.94, 7.85};
  DataVector t;
  t.set_data_ref(data);
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
  t_copy.set_data_ref(t);
  test_move_semantics(t, t_copy);
  DataVector t_move_assignment = std::move(t_copy);
  CHECK(not t_move_assignment.is_owning());
  DataVector t_move_constructor = std::move(t_move_assignment);
  CHECK(not t_move_constructor.is_owning());
  {
    DataVector data_2{1.43, 2.83, 3.94, 7.85};
    DataVector data_2_ref;
    data_2_ref.set_data_ref(data_2);
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
    owned_data = data_2_ref;
    CHECK(owned_data[0] == 2.43);
    CHECK(owned_data[1] == 3.83);
    CHECK(owned_data[2] == 4.94);
    CHECK(owned_data[3] == 8.85);
    // Test operator!=
    CHECK_FALSE(owned_data != data_2_ref);
  }
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] TEST_CASE("Unit.DataStructures.DataVector.ref_diff_size",
                       "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DataVector data{1.43, 2.83, 3.94, 7.85};
  DataVector data_ref;
  data_ref.set_data_ref(data);
  DataVector data2{1.43, 2.83, 3.94};
  data_ref = data2;
  ERROR("Bad end");
#endif
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] TEST_CASE("Unit.DataStructures.DataVector.move_ref_diff_size",
                       "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DataVector data{1.43, 2.83, 3.94, 7.85};
  DataVector data_ref;
  data_ref.set_data_ref(data);
  DataVector data2{1.43, 2.83, 3.94};
  data_ref = std::move(data2);
  ERROR("Bad end");
#endif
}

namespace {
template <typename T1, typename T2>
void check_vectors(const T1& t1, const T2& t2) {
  CHECK(t1.size() == t2.size());
  for (size_t i = 0; i < t1.size(); ++i) {
    CHECK(t1[i] == Approx(t2[i]).epsilon(1e-12));
  }
}
}  // namespace

TEST_CASE("Unit.DataStructures.DataVector.Math", "[Unit][DataStructures]") {
  constexpr size_t num_pts = 19;
  DataVector val{1, 2, 3, -4, 8, 12, -14};
  DataVector nine(num_pts, 9.0);
  DataVector one(num_pts, 1.0);

  // Test unary minus
  check_vectors(-nine, DataVector(num_pts, -9.0));

  check_vectors(nine + 2.0, DataVector(num_pts, 11.0));
  check_vectors(2.0 + nine, DataVector(num_pts, 11.0));
  check_vectors(nine - 2.0, DataVector(num_pts, 7.0));
  check_vectors(2.0 - nine, DataVector(num_pts, -7.0));
  check_vectors(nine + nine, DataVector(num_pts, 18.0));
  check_vectors(nine + (one * nine), DataVector(num_pts, 18.0));
  check_vectors((one * nine) + nine, DataVector(num_pts, 18.0));
  check_vectors(nine - nine, DataVector(num_pts, 0.0));
  check_vectors(nine - (one * nine), DataVector(num_pts, 0.0));
  check_vectors((one * nine) - nine, DataVector(num_pts, 0.0));

  CHECK(-14 == min(val));
  CHECK(12 == max(val));
  check_vectors(DataVector{1, 2, 3, 4, 8, 12, 14}, abs(val));

  check_vectors(DataVector(num_pts, 81.0), nine * nine);
  check_vectors(DataVector(num_pts, 81.0), nine * (nine * one));
  check_vectors(DataVector(num_pts, 81.0), (nine * nine) * one);
  check_vectors(DataVector(num_pts, 81.0), 9.0 * nine);
  check_vectors(DataVector(num_pts, 81.0), nine * 9.0);
  check_vectors(DataVector(num_pts, 1.0), nine / 9.0);
  check_vectors(DataVector(num_pts, 1.0), nine / (one * 9.0));
  check_vectors(DataVector(num_pts, 9.0), (one * nine) / one);

  check_vectors(sqrt(nine), DataVector(num_pts, 3.0));
  check_vectors(invsqrt(nine), DataVector(num_pts, 1.0 / 3.0));
  DataVector eight(num_pts, 8.0);
  check_vectors(cbrt(eight), DataVector(num_pts, 2.0));
  check_vectors(invcbrt(eight), DataVector(num_pts, 0.5));
  check_vectors(DataVector(num_pts, 81.0), pow(nine, 2));

  DataVector dummy(nine * nine * 1.0);
  check_vectors(DataVector(num_pts, 81.0), dummy);
  check_vectors(DataVector(num_pts, 81.0), pow<2>(nine));
  check_vectors(DataVector(num_pts, 81.0), pow<2>(nine * one));
  check_vectors(DataVector(num_pts, 1.0 / 81.0),
                DataVector(num_pts, 1.0) / pow<2>(nine));

  check_vectors(DataVector(num_pts, exp(9.0)), exp(nine));
  check_vectors(DataVector(num_pts, exp2(9.0)), exp2(nine));
  check_vectors(DataVector(num_pts, pow<9>(10.0)), DataVector(exp10(nine)));
  check_vectors(DataVector(num_pts, log(9.0)), log(nine));
  check_vectors(DataVector(num_pts, log2(9.0)), log2(nine));
  check_vectors(DataVector(num_pts, log10(9.0)), log10(nine));

  DataVector point_nine(num_pts, 0.9);
  check_vectors(DataVector(num_pts, sin(9.0)), sin(nine));
  check_vectors(DataVector(num_pts, cos(9.0)), cos(nine));
  check_vectors(DataVector(num_pts, tan(9.0)), tan(nine));
  check_vectors(DataVector(num_pts, asin(0.9)), asin(point_nine));
  check_vectors(DataVector(num_pts, acos(0.9)), acos(point_nine));
  check_vectors(DataVector(num_pts, atan(0.9)), atan(point_nine));

  check_vectors(DataVector(num_pts, sinh(9.0)), sinh(nine));
  check_vectors(DataVector(num_pts, cosh(9.0)), cosh(nine));
  check_vectors(DataVector(num_pts, tanh(9.0)), tanh(nine));
  check_vectors(DataVector(num_pts, asinh(9.0)), asinh(nine));
  check_vectors(DataVector(num_pts, acosh(9.0)), acosh(nine));
  check_vectors(DataVector(num_pts, atanh(0.9)), atanh(point_nine));

  check_vectors(DataVector(num_pts, erf(0.9)), erf(point_nine));
  check_vectors(DataVector(num_pts, erfc(0.9)), erfc(point_nine));

  // Test assignment
  DataVector test_81(num_pts, -1.0);
  test_81 = nine * nine;
  check_vectors(DataVector(num_pts, 81.0), test_81);
  CHECK(test_81.is_owning());
  DataVector test_81_ref(test_81.data(), test_81.size());
  test_81_ref = 0.0;
  test_81 = 0.0;
  test_81_ref = nine * nine;
  check_vectors(DataVector(num_pts, 81.0), test_81);
  CHECK(test_81.is_owning());
  check_vectors(DataVector(num_pts, 81.0), test_81_ref);
  CHECK_FALSE(test_81_ref.is_owning());
  DataVector second_81(num_pts);
  second_81 = test_81;
  check_vectors(DataVector(num_pts, 81.0), second_81);
  CHECK(second_81.is_owning());
  test_81_ref = 0.0;
  check_vectors(DataVector(num_pts, 0.0), test_81_ref);
  test_81_ref = second_81;
  check_vectors(DataVector(num_pts, 81.0), test_81_ref);
  second_81 = 0.0;
  test_81_ref.set_data_ref(test_81);
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
  test_081.set_data_ref(test_81);
  check_vectors(DataVector(num_pts, 81.0), test_081);
  CHECK_FALSE(test_081.is_owning());
  test_081 = square(point_nine);
  check_vectors(DataVector(num_pts, 0.81), test_081);
  CHECK_FALSE(test_081.is_owning());

  DataVector test_assignment(num_pts, 7.0);
  test_assignment += nine;
  check_vectors(DataVector(num_pts, 16.0), test_assignment);
  test_assignment += 3.0;
  check_vectors(DataVector(num_pts, 19.0), test_assignment);
  test_assignment += (nine * nine);
  check_vectors(DataVector(num_pts, 100.0), test_assignment);

  test_assignment = 7.0;
  test_assignment -= nine;
  check_vectors(DataVector(num_pts, -2.0), test_assignment);
  test_assignment -= 3.0;
  check_vectors(DataVector(num_pts, -5.0), test_assignment);
  test_assignment -= nine * nine;
  check_vectors(DataVector(num_pts, -86.0), test_assignment);

  test_assignment = 2.0;
  test_assignment *= 3.0;
  check_vectors(DataVector(num_pts, 6.0), test_assignment);
  test_assignment *= nine;
  check_vectors(DataVector(num_pts, 54.0), test_assignment);
  test_assignment = 1.0;
  test_assignment *= nine * nine;
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
  x += sqrt(x);
  check_vectors(DataVector(num_pts, 6.0), x);
  x -= sqrt(x - 2.0);
  check_vectors(DataVector(num_pts, 4.0), x);
  x = sqrt(x);
  check_vectors(DataVector(num_pts, 2.0), x);
  x *= x;
  check_vectors(DataVector(num_pts, 4.0), x);
  x /= x;
  check_vectors(DataVector(num_pts, 1.0), x);
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] TEST_CASE("Unit.DataStructures.DataVector.ExpressionAssignError",
                       "[Unit][DataStructures]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DataVector one(10, 1.0);
  DataVector one_ref(one.data(), one.size());
  DataVector one_b(2, 1.0);
  one_ref = (one_b * one_b);
  ERROR("Bad end");
#endif
}
