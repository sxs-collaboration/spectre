// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace VariablesTestTags_detail {
struct vector {
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
};

struct scalar {
  using type = Scalar<DataVector>;
};

struct scalar2 {
  using type = Scalar<DataVector>;
};
}  // namespace VariablesTestTags_detail

static_assert(
    std::is_nothrow_move_constructible<
        Variables<tmpl::list<VariablesTestTags_detail::scalar,
                             VariablesTestTags_detail::vector>>>::value,
    "Missing move semantics in Variables.");

TEST_CASE("Unit.DataStructures.Variables", "[DataStructures][Unit]") {
  Variables<tmpl::list<VariablesTestTags_detail::vector,
                       VariablesTestTags_detail::scalar,
                       VariablesTestTags_detail::scalar2>>
      v(1, -3.0);

  CHECK(v.size() ==
        v.number_of_grid_points() * v.number_of_independent_components);

  CHECK(1 == v.number_of_grid_points());
  CHECK(5 == v.size());

  auto& vector_in_v = v.get<VariablesTestTags_detail::vector>();
  CHECK(-3.0 == v.data()[0]);
  CHECK(-3.0 == vector_in_v.get(0)[0]);

  tnsr::I<DataVector, 3, Frame::Grid> another_vector(1_st, -5.0);

  CHECK(-5.0 == another_vector.get(0)[0]);

  vector_in_v = another_vector;

  CHECK(-5.0 == v.data()[0]);
  CHECK(-5.0 == vector_in_v.get(0)[0]);

  vector_in_v = tnsr::I<DataVector, 3, Frame::Grid>{1_st, -4.0};

  CHECK(-4.0 == v.data()[0]);
  CHECK(-4.0 == v.data()[1]);
  CHECK(-4.0 == v.data()[2]);
  CHECK(-4.0 == vector_in_v.get(0)[0]);
  CHECK(-4.0 == vector_in_v.get(1)[0]);
  CHECK(-4.0 == vector_in_v.get(2)[0]);

  const auto& kvector_in_v = v.get<VariablesTestTags_detail::vector>();
  CHECK(kvector_in_v.get(0)[0] == -4.0);

  Variables<tmpl::list<VariablesTestTags_detail::vector,
                       VariablesTestTags_detail::scalar,
                       VariablesTestTags_detail::scalar2>>
      v2(1, -3.0);
  CHECK(v != v2);
  v2 = v;
  CHECK(v2 == v);

  Variables<tmpl::list<VariablesTestTags_detail::vector,
                       VariablesTestTags_detail::scalar,
                       VariablesTestTags_detail::scalar2>>
      v3;
  CHECK(v3.size() == 0);
  CHECK(v3.number_of_grid_points() == 0);
  v3 = v2;
  CHECK(v2 == v3);

  const std::string expected_output =
      "--Symmetry:  (1)\n"
      "--Types:     (Spatial)\n"
      "--Dims:      (3)\n"
      "--Locations: (Up)\n"
      "--Frames:    (Grid)\n"
      " T(0)=(-4)\n"
      "     Multiplicity: 1 Index: 0\n"
      " T(1)=(-4)\n"
      "     Multiplicity: 1 Index: 1\n"
      " T(2)=(-4)\n"
      "     Multiplicity: 1 Index: 2\n"
      "--Symmetry:  ()\n"
      "--Types:     ()\n"
      "--Dims:      ()\n"
      "--Locations: ()\n"
      "--Frames:    ()\n"
      " T(0)=(-3)\n"
      "     Multiplicity: 1 Index: 0\n"
      "--Symmetry:  ()\n"
      "--Types:     ()\n"
      "--Dims:      ()\n"
      "--Locations: ()\n"
      "--Frames:    ()\n"
      " T(0)=(-3)\n"
      "     Multiplicity: 1 Index: 0";
  CHECK(get_output(v) == expected_output);

  // Check self-assignment
  v = v;
  CHECK(v == v2);
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] TEST_CASE("Unit.DataStructures.Variables.BadCopy",
                       "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Variables<tmpl::list<VariablesTestTags_detail::vector,
                       VariablesTestTags_detail::scalar,
                       VariablesTestTags_detail::scalar2>>
      v(1, -3.0);
  auto& vector_in_v = v.get<VariablesTestTags_detail::vector>();
  vector_in_v = tnsr::I<DataVector, 3, Frame::Grid>{10_st, -4.0};
  ERROR("Bad test end");
#endif
}

TEST_CASE("Unit.DataStructures.Variables.Move", "[DataStructures][Unit]") {
  Variables<tmpl::list<VariablesTestTags_detail::vector>> x(1, -2.0),
      z(2, -3.0);
  CHECK(&z.template get<VariablesTestTags_detail::vector>()[0][0] == z.data());
  Variables<tmpl::list<VariablesTestTags_detail::vector>> y = std::move(x);
  x = std::move(z);
  CHECK(
      (x == Variables<tmpl::list<VariablesTestTags_detail::vector>>{2, -3.0}));
  CHECK(&x.template get<VariablesTestTags_detail::vector>()[0][0] == x.data());

// Intentionally testing self-move
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif  // defined(__clang__)
  x = std::move(x);
#ifdef __clang__
#pragma GCC diagnostic pop
#endif  // defined(__clang__)
  CHECK(
      (x == Variables<tmpl::list<VariablesTestTags_detail::vector>>{2, -3.0}));
  CHECK(&x.template get<VariablesTestTags_detail::vector>()[0][0] == x.data());
}

namespace {
template <typename T1, typename VT, bool VF>
void check_vectors(const Variables<T1>& t1, const blaze::Vector<VT, VF>& t2) {
  CHECK(t1.size() == (~t2).size());
  for (size_t i = 0; i < t1.size(); ++i) {
    // We've removed the subscript operator so people don't try to use that
    // and as a result we need to use the data() member function
    CHECK(t1.data()[i] == Approx((~t2)[i]).epsilon(1e-14));
  }
}

template <typename T1, typename T2>
void check_vectors(const Variables<T1>& t1, const Variables<T2>& t2) {
  CHECK(t1.size() == t2.size());
  for (size_t i = 0; i < t1.size(); ++i) {
    // We've removed the subscript operator so people don't try to use that
    // and as a result we need to use the data() member function
    CHECK(t1.data()[i] == Approx(t2.data()[i]).epsilon(1e-14));
  }
}
}  // namespace

TEST_CASE("Unit.DataStructures.Variables.Math", "[DataStructures][Unit]") {
  using test_variable_type =
      Variables<tmpl::list<VariablesTestTags_detail::vector,
                           VariablesTestTags_detail::scalar,
                           VariablesTestTags_detail::scalar2>>;
  test_variable_type three(1, 3.0);
  check_vectors(test_variable_type{1, 6}, 2.0 * three);
  check_vectors(test_variable_type{1, 6}, three * 2.0);
  check_vectors(test_variable_type{1, 1.5}, three / 2.0);
  check_vectors(test_variable_type{1, 12}, three + three + three + three);
  check_vectors(test_variable_type{1, 9}, three + (three + three));

  check_vectors(test_variable_type{1, -6}, three - three - three - three);
  check_vectors(test_variable_type{1, 3}, three - (three - three));
  check_vectors(test_variable_type{1, 3}, three - three + three);
  check_vectors(test_variable_type{1, 3}, three + three - three);

  test_variable_type test_assignment(three * 1.0);
  test_assignment += test_variable_type{1, 3};
  check_vectors(test_variable_type{1, 6}, test_assignment);
  test_assignment -= test_variable_type{1, 2};
  check_vectors(test_variable_type{1, 4}, test_assignment);
  test_assignment *= 0.25;
  check_vectors(test_variable_type{1, 1.0}, test_assignment);
  test_assignment /= 0.1;
  check_vectors(test_variable_type{1, 10.0}, test_assignment);

  test_assignment += test_variable_type{1, 3} * 3.0;
  check_vectors(test_variable_type{1, 19.0}, test_assignment);
  test_assignment -= test_variable_type{1, 3} * 2.0;
  check_vectors(test_variable_type{1, 13.0}, test_assignment);

  test_variable_type test_assignment2(1, 0.0);
  test_assignment2 = test_assignment * 1.0;
  CHECK(test_assignment2 == test_assignment);
}

TEST_CASE("Unit.DataStructures.Variables.Serialization",
          "[DataStructures][Unit]") {
  Variables<tmpl::list<VariablesTestTags_detail::vector>> v(1, -3.0);
  CHECK(v == serialize_and_deserialize(v));
}
