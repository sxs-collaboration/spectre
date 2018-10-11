// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "ErrorHandling/Error.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Variables

namespace VariablesTestTags_detail {
/// [simple_variables_tag]
struct vector : db::SimpleTag {
  static std::string name() noexcept { return "vector"; }
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
};
/// [simple_variables_tag]

struct scalar : db::SimpleTag {
  static std::string name() noexcept { return "scalar"; }
  using type = Scalar<DataVector>;
};

struct scalar2 : db::SimpleTag {
  static std::string name() noexcept { return "scalar2"; }
  using type = Scalar<DataVector>;
};

/// [prefix_variables_tag]
template <class Tag>
struct PrefixTag0 : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "PrefixTag0"; }
};
/// [prefix_variables_tag]

template <class Tag>
struct PrefixTag1 : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "PrefixTag1"; }
};

template <class Tag>
struct PrefixTag2 : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "PrefixTag2"; }
};

template <class Tag>
struct PrefixTag3 : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "PrefixTag3"; }
};
}  // namespace VariablesTestTags_detail

static_assert(
    std::is_nothrow_move_constructible<
        Variables<tmpl::list<VariablesTestTags_detail::scalar,
                             VariablesTestTags_detail::vector>>>::value,
    "Missing move semantics in Variables.");

SPECTRE_TEST_CASE("Unit.DataStructures.Variables", "[DataStructures][Unit]") {
  Variables<tmpl::list<VariablesTestTags_detail::vector,
                       VariablesTestTags_detail::scalar,
                       VariablesTestTags_detail::scalar2>>
      v(1, -3.0);
  decltype(v) v_initialize;
  v_initialize.initialize(1, -3.0);
  CHECK(v == v_initialize);

  CHECK(v.size() ==
        v.number_of_grid_points() * v.number_of_independent_components);

  CHECK(1 == v.number_of_grid_points());
  CHECK(5 == v.size());

  auto& vector_in_v = get<VariablesTestTags_detail::vector>(v);
  // clang-tidy: do not use pointer arithmetic
  CHECK(-3.0 == v.data()[0]);  // NOLINT
  CHECK(-3.0 == vector_in_v.get(0)[0]);

  tnsr::I<DataVector, 3, Frame::Grid> another_vector(1_st, -5.0);

  CHECK(-5.0 == another_vector.get(0)[0]);

  vector_in_v = another_vector;

  // clang-tidy: do not use pointer arithmetic
  CHECK(-5.0 == v.data()[0]);  // NOLINT
  CHECK(-5.0 == vector_in_v.get(0)[0]);

  vector_in_v = tnsr::I<DataVector, 3, Frame::Grid>{1_st, -4.0};

  // clang-tidy: do not use pointer arithmetic
  CHECK(-4.0 == v.data()[0]);  // NOLINT
  CHECK(-4.0 == v.data()[1]);  // NOLINT
  CHECK(-4.0 == v.data()[2]);  // NOLINT
  CHECK(-4.0 == vector_in_v.get(0)[0]);
  CHECK(-4.0 == vector_in_v.get(1)[0]);
  CHECK(-4.0 == vector_in_v.get(2)[0]);

  const auto& kvector_in_v = get<VariablesTestTags_detail::vector>(v);
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
      "vector:\n"
      "T(0)=(-4)\n"
      "T(1)=(-4)\n"
      "T(2)=(-4)\n\n"
      "scalar:\n"
      "T()=(-3)\n\n"
      "scalar2:\n"
      "T()=(-3)";
  CHECK(get_output(v) == expected_output);

  Variables<tmpl::list<>> empty_vars;
  CHECK(get_output(empty_vars) == "Variables is empty!");

  // Check self-assignment
#ifndef __APPLE__
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
#endif  // defined(__clang__) && __clang_major__ > 6
#endif  // ! __APPLE__
  v = v;
#ifndef __APPLE__
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic pop
#endif  // defined(__clang__) && __clang_major__ > 6
#endif  // ! __APPLE__
  CHECK(v == v2);

  CHECK(
      Tags::Variables<tmpl::list<VariablesTestTags_detail::vector,
                                 VariablesTestTags_detail::scalar,
                                 VariablesTestTags_detail::scalar2>>::name() ==
      "Variables(vector,scalar,scalar2)");
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.DataStructures.Variables.BadCopy",
                               "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Variables<tmpl::list<VariablesTestTags_detail::vector,
                       VariablesTestTags_detail::scalar,
                       VariablesTestTags_detail::scalar2>>
      v(1, -3.0);
  auto& vector_in_v = get<VariablesTestTags_detail::vector>(v);
  vector_in_v = tnsr::I<DataVector, 3, Frame::Grid>{10_st, -4.0};
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

SPECTRE_TEST_CASE("Unit.DataStructures.Variables.Move",
                  "[DataStructures][Unit]") {
  Variables<tmpl::list<VariablesTestTags_detail::vector>> x(1, -2.0),
      z(2, -3.0);
  CHECK(&get<VariablesTestTags_detail::vector>(z)[0][0] == z.data());
  Variables<tmpl::list<VariablesTestTags_detail::vector>> y = std::move(x);
  x = std::move(z);
  CHECK(
      (x == Variables<tmpl::list<VariablesTestTags_detail::vector>>{2, -3.0}));
  CHECK(&get<VariablesTestTags_detail::vector>(x)[0][0] == x.data());

// Intentionally testing self-move
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif  // defined(__clang__)
  x = std::move(x);
#ifdef __clang__
#pragma GCC diagnostic pop
#endif  // defined(__clang__)
  // clang-tidy: false positive 'x' used after it was moved
  CHECK((x ==  // NOLINT
         Variables<tmpl::list<VariablesTestTags_detail::vector>>{2, -3.0}));
  CHECK(&get<VariablesTestTags_detail::vector>(x)[0][0] == x.data());
}

namespace {
template <typename T1, typename VT, bool VF>
void check_vectors(const Variables<T1>& t1, const blaze::Vector<VT, VF>& t2) {
  CHECK(t1.size() == (~t2).size());
  for (size_t i = 0; i < t1.size(); ++i) {
    // We've removed the subscript operator so people don't try to use that
    // and as a result we need to use the data() member function
    // clang-tidy: do not use pointer arithmetic
    CHECK(t1.data()[i] == approx((~t2)[i]));  // NOLINT
  }
}

template <typename T1, typename T2>
void check_vectors(const Variables<T1>& t1, const Variables<T2>& t2) {
  CHECK(t1.size() == t2.size());
  for (size_t i = 0; i < t1.size(); ++i) {
    // We've removed the subscript operator so people don't try to use that
    // and as a result we need to use the data() member function
    // clang-tidy: do not use pointer arithmetic
    CHECK(t1.data()[i] == approx(t2.data()[i]));  // NOLINT
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Variables.Math",
                  "[DataStructures][Unit]") {
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

  // clang-tidy: point sides of overloaded operator are equivalent
  check_vectors(test_variable_type{1, -6},
                three - three - three - three);                      // NOLINT
  check_vectors(test_variable_type{1, 3}, three - (three - three));  // NOLINT
  check_vectors(test_variable_type{1, 3}, three - three + three);    // NOLINT
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

  const auto check_components = [](const auto& variables,
                                   const DataVector& datavector) {
    tmpl::for_each<typename std::decay_t<decltype(variables)>::tags_list>(
        [&variables, &datavector](auto tag) {
          using Tag = tmpl::type_from<decltype(tag)>;
          for (const auto& component : get<Tag>(variables)) {
            CHECK(component == datavector);
          }
        });
  };
  const DataVector dv{1., 2., 3., 4.};
  test_variable_type test_datavector_math(4, 2.);
  test_datavector_math *= dv;
  check_components(test_datavector_math, {2., 4., 6., 8.});
  check_components(test_datavector_math * dv, {2., 8., 18., 32.});
  check_components(dv * test_datavector_math, {2., 8., 18., 32.});
  test_datavector_math *= dv;
  check_components(test_datavector_math, {2., 8., 18., 32.});
  check_components(test_datavector_math / dv, {2., 4., 6., 8.});
  test_datavector_math /= dv;
  check_components(test_datavector_math, {2., 4., 6., 8.});
}

SPECTRE_TEST_CASE("Unit.DataStructures.Variables.PrefixSemantics",
                  "[DataStructures][Unit]") {
  using variable_type =
      Variables<tmpl::list<VariablesTestTags_detail::vector,
                           VariablesTestTags_detail::scalar,
                           VariablesTestTags_detail::scalar2>>;
  using prefix_variable_type = Variables<tmpl::list<
      VariablesTestTags_detail::PrefixTag0<VariablesTestTags_detail::vector>,
      VariablesTestTags_detail::PrefixTag0<VariablesTestTags_detail::scalar>,
      VariablesTestTags_detail::PrefixTag0<VariablesTestTags_detail::scalar2>>>;
  prefix_variable_type prefix_vars_0(4, 8.);
  prefix_variable_type prefix_vars_1(4, 6.);
  variable_type vars_0(prefix_vars_0);
  variable_type vars_1(std::move(prefix_vars_1));

  const auto check_variables = [](const auto& lvars_0, const auto& lvars_1) {
    tmpl::for_each<typename std::decay_t<decltype(lvars_0)>::tags_list>(
        [&lvars_0, &lvars_1](auto tag) {
          using Tag = tmpl::type_from<decltype(tag)>;
          for (const auto& component :
               boost::combine(get<Tag>(lvars_0), get<Tag>(lvars_1))) {
            CHECK(boost::get<0>(component) == boost::get<1>(component));
          }
        });
  };

  check_variables(vars_0, variable_type(4, 8.0));
  check_variables(vars_1, variable_type(4, 6.0));

  prefix_variable_type prefix_vars_2(4, 4.);
  prefix_variable_type prefix_vars_3(4, 2.);
  vars_0 = prefix_vars_2;
  check_variables(vars_0, variable_type(4, 4.0));
  vars_0 = std::move(prefix_vars_3);
  check_variables(vars_0, variable_type(4, 2.0));
}

SPECTRE_TEST_CASE("Unit.DataStructures.Variables.PrefixMath",
                  "[DataStructures][Unit]") {
  using prefix0_variable_type = Variables<tmpl::list<
      VariablesTestTags_detail::PrefixTag0<VariablesTestTags_detail::vector>,
      VariablesTestTags_detail::PrefixTag0<VariablesTestTags_detail::scalar>,
      VariablesTestTags_detail::PrefixTag0<VariablesTestTags_detail::scalar2>>>;
  using prefix1_variable_type = Variables<tmpl::list<
      VariablesTestTags_detail::PrefixTag1<VariablesTestTags_detail::vector>,
      VariablesTestTags_detail::PrefixTag1<VariablesTestTags_detail::scalar>,
      VariablesTestTags_detail::PrefixTag1<VariablesTestTags_detail::scalar2>>>;
  using prefix2_variable_type = Variables<tmpl::list<
      VariablesTestTags_detail::PrefixTag2<VariablesTestTags_detail::vector>,
      VariablesTestTags_detail::PrefixTag2<VariablesTestTags_detail::scalar>,
      VariablesTestTags_detail::PrefixTag2<VariablesTestTags_detail::scalar2>>>;
  using prefix3_variable_type = Variables<tmpl::list<
      VariablesTestTags_detail::PrefixTag3<VariablesTestTags_detail::PrefixTag2<
          VariablesTestTags_detail::vector>>,
      VariablesTestTags_detail::PrefixTag3<VariablesTestTags_detail::PrefixTag2<
          VariablesTestTags_detail::scalar>>,
      VariablesTestTags_detail::PrefixTag3<VariablesTestTags_detail::PrefixTag2<
          VariablesTestTags_detail::scalar2>>>>;

  prefix0_variable_type three_0(1, 3.0);
  prefix1_variable_type three_1(1, 3.0);
  prefix2_variable_type three_2(1, 3.0);
  prefix3_variable_type three_3(1, 3.0);

  check_vectors(prefix3_variable_type{1, 6.0}, 2.0 * three_0);
  check_vectors(prefix3_variable_type{1, 6.0}, three_0 * 2.0);
  check_vectors(prefix3_variable_type{1, 1.5}, three_0 / 2.0);
  check_vectors(prefix3_variable_type{1, 12.0},
                three_0 + three_1 + three_2 + three_3);
  check_vectors(prefix3_variable_type{1, 12.0},
                (three_0 + three_1) + (three_2 + three_3));
  check_vectors(prefix3_variable_type{1, 9.0}, three_0 + (three_1 + three_2));

  // clang-tidy: point sides of overloaded operator are equivalent
  check_vectors(prefix3_variable_type{1, -6.0},
                three_0 - three_1 - three_2 - three_3);  // NOLINT
  check_vectors(prefix3_variable_type{1, 6.0},
                three_0 - (three_1 - three_2 - three_3));  // NOLINT
  check_vectors(prefix3_variable_type{1, 0.0},
                (three_0 - three_1) - (three_2 - three_3));  // NOLINT
  check_vectors(prefix3_variable_type{1, 3.0},
                three_0 - (three_1 - three_2));  // NOLINT
  check_vectors(prefix3_variable_type{1, 3.0},
                three_0 - three_1 + three_2);  // NOLINT
  check_vectors(prefix3_variable_type{1, 3.0}, three_0 + three_1 - three_2);

  prefix0_variable_type test_assignment(three_0 * 1.0);
  test_assignment += prefix1_variable_type{1, 3};
  check_vectors(prefix0_variable_type{1, 6}, test_assignment);
  test_assignment -= prefix1_variable_type{1, 2};
  check_vectors(prefix0_variable_type{1, 4}, test_assignment);
  test_assignment *= 0.25;
  check_vectors(prefix0_variable_type{1, 1.0}, test_assignment);
  test_assignment /= 0.1;
  check_vectors(prefix0_variable_type{1, 10.0}, test_assignment);

  test_assignment += prefix1_variable_type{1, 3} * 3.0;
  check_vectors(prefix0_variable_type{1, 19.0}, test_assignment);
  test_assignment -= prefix1_variable_type{1, 3} * 2.0;
  check_vectors(prefix0_variable_type{1, 13.0}, test_assignment);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Variables.Serialization",
                  "[DataStructures][Unit]") {
  Variables<tmpl::list<VariablesTestTags_detail::vector>> v(1, -3.0);
  test_serialization(v);
  auto tuple_of_v = std::make_tuple(
      Variables<tmpl::list<VariablesTestTags_detail::vector>>{1, -4.0});
  test_serialization(tuple_of_v);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Variables.assign_subset",
                  "[DataStructures][Unit]") {
  constexpr size_t size = 3;
  const auto first_test = [&size](const auto& vars_subset0,
                                  const auto& vars_subset1) noexcept {
    Variables<tmpl::list<VariablesTestTags_detail::vector,
                         VariablesTestTags_detail::scalar>>
        vars_set0{size, 3.0};
    CHECK(get<VariablesTestTags_detail::vector>(vars_set0) ==
          db::item_type<VariablesTestTags_detail::vector>(size, 3.0));
    CHECK(get<VariablesTestTags_detail::scalar>(vars_set0) ==
          db::item_type<VariablesTestTags_detail::scalar>(size, 3.0));
    vars_set0.assign_subset(vars_subset0);
    CHECK(get<VariablesTestTags_detail::vector>(vars_set0) ==
          db::item_type<VariablesTestTags_detail::vector>(size, 8.0));
    CHECK(get<VariablesTestTags_detail::scalar>(vars_set0) ==
          db::item_type<VariablesTestTags_detail::scalar>(size, 3.0));
    vars_set0.assign_subset(vars_subset1);
    CHECK(get<VariablesTestTags_detail::vector>(vars_set0) ==
          db::item_type<VariablesTestTags_detail::vector>(size, 8.0));
    CHECK(get<VariablesTestTags_detail::scalar>(vars_set0) ==
          db::item_type<VariablesTestTags_detail::scalar>(size, 4.0));
  };

  first_test(
      Variables<tmpl::list<VariablesTestTags_detail::vector>>(size, 8.0),
      Variables<tmpl::list<VariablesTestTags_detail::scalar>>(size, 4.0));
  first_test(tuples::TaggedTuple<VariablesTestTags_detail::vector>(
                 VariablesTestTags_detail::vector::type{size, 8.0}),
             tuples::TaggedTuple<VariablesTestTags_detail::scalar>(
                 VariablesTestTags_detail::scalar::type{size, 4.0}));

  const auto second_test = [&size](const auto& vars_subset0,
                                   const auto& vars_subset1) noexcept {
    Variables<tmpl::list<VariablesTestTags_detail::vector,
                         VariablesTestTags_detail::scalar,
                         VariablesTestTags_detail::scalar2>>
        vars_set1(size, 3.0);
    CHECK(get<VariablesTestTags_detail::vector>(vars_set1) ==
          db::item_type<VariablesTestTags_detail::vector>(size, 3.0));
    CHECK(get<VariablesTestTags_detail::scalar>(vars_set1) ==
          db::item_type<VariablesTestTags_detail::scalar>(size, 3.0));
    CHECK(get<VariablesTestTags_detail::scalar2>(vars_set1) ==
          db::item_type<VariablesTestTags_detail::scalar2>(size, 3.0));
    vars_set1.assign_subset(vars_subset0);
    CHECK(get<VariablesTestTags_detail::vector>(vars_set1) ==
          db::item_type<VariablesTestTags_detail::vector>(size, 8.0));
    CHECK(get<VariablesTestTags_detail::scalar>(vars_set1) ==
          db::item_type<VariablesTestTags_detail::scalar>(size, 3.0));
    CHECK(get<VariablesTestTags_detail::scalar2>(vars_set1) ==
          db::item_type<VariablesTestTags_detail::scalar2>(size, 3.0));
    vars_set1.assign_subset(vars_subset1);
    CHECK(get<VariablesTestTags_detail::vector>(vars_set1) ==
          db::item_type<VariablesTestTags_detail::vector>(size, 8.0));
    CHECK(get<VariablesTestTags_detail::scalar>(vars_set1) ==
          db::item_type<VariablesTestTags_detail::scalar>(size, 4.0));
    CHECK(get<VariablesTestTags_detail::scalar2>(vars_set1) ==
          db::item_type<VariablesTestTags_detail::scalar2>(size, 3.0));
  };

  second_test(
      Variables<tmpl::list<VariablesTestTags_detail::vector>>(size, 8.0),
      Variables<tmpl::list<VariablesTestTags_detail::scalar>>(size, 4.0));
  second_test(tuples::TaggedTuple<VariablesTestTags_detail::vector>(
                  VariablesTestTags_detail::vector::type{size, 8.0}),
              tuples::TaggedTuple<VariablesTestTags_detail::scalar>(
                  VariablesTestTags_detail::scalar::type{size, 4.0}));

  const auto third_test = [&size](const auto& vars_subset0) noexcept {
    Variables<tmpl::list<VariablesTestTags_detail::vector>> vars_set2(size,
                                                                      -7.0);
    CHECK(get<VariablesTestTags_detail::vector>(vars_set2) ==
          db::item_type<VariablesTestTags_detail::vector>(size, -7.0));
    vars_set2.assign_subset(vars_subset0);
    CHECK(get<VariablesTestTags_detail::vector>(vars_set2) ==
          db::item_type<VariablesTestTags_detail::vector>(size, 8.0));
  };

  third_test(
      Variables<tmpl::list<VariablesTestTags_detail::vector>>(size, 8.0));
  third_test(tuples::TaggedTuple<VariablesTestTags_detail::vector>(
      VariablesTestTags_detail::vector::type{size, 8.0}));
}

SPECTRE_TEST_CASE("Unit.DataStructures.Variables.SliceVariables",
                  "[DataStructures][Unit]") {
  Variables<tmpl::list<VariablesTestTags_detail::vector>> vars(24, 0.);
  const size_t x_extents = 2, y_extents = 3, z_extents = 4,
               vec_size = VariablesTestTags_detail::vector::type::size();
  Index<3> extents(x_extents, y_extents, z_extents);
  for (size_t s = 0; s < vars.size(); ++s) {
    // clang-tidy: do not use pointer arithmetic
    vars.data()[s] = s;  // NOLINT
  }
  Variables<tmpl::list<VariablesTestTags_detail::vector>>
      expected_vars_sliced_in_x(y_extents * z_extents, 0.),
      expected_vars_sliced_in_y(x_extents * z_extents, 0.),
      expected_vars_sliced_in_z(x_extents * y_extents, 0.);
  const size_t x_offset = 1, y_offset = 2, z_offset = 1;

  for (size_t s = 0; s < expected_vars_sliced_in_x.size(); ++s) {
    // clang-tidy: do not use pointer arithmetic
    expected_vars_sliced_in_x.data()[s] = x_offset + s * x_extents;  // NOLINT
  }
  for (size_t i = 0; i < vec_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t z = 0; z < z_extents; ++z) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_y
            .data()[x + x_extents * (z + z_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y_offset + z * y_extents);
      }
    }
  }
  for (size_t i = 0; i < vec_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t y = 0; y < y_extents; ++y) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_z
            .data()[x + x_extents * (y + y_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y + y_extents * z_offset);
      }
    }
  }
  CHECK(data_on_slice(vars, extents, 0, x_offset) == expected_vars_sliced_in_x);
  CHECK(data_on_slice(vars, extents, 1, y_offset) == expected_vars_sliced_in_y);
  CHECK(data_on_slice(vars, extents, 2, z_offset) == expected_vars_sliced_in_z);

  CHECK(
      data_on_slice<VariablesTestTags_detail::vector>(
          extents, 0, x_offset, get<VariablesTestTags_detail::vector>(vars)) ==
      expected_vars_sliced_in_x);
  CHECK(
      data_on_slice<VariablesTestTags_detail::vector>(
          extents, 1, y_offset, get<VariablesTestTags_detail::vector>(vars)) ==
      expected_vars_sliced_in_y);
  CHECK(
      data_on_slice<VariablesTestTags_detail::vector>(
          extents, 2, z_offset, get<VariablesTestTags_detail::vector>(vars)) ==
      expected_vars_sliced_in_z);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Variables.add_slice_to_data",
                  "[DataStructures][Unit]") {
  using Vector = VariablesTestTags_detail::vector::type;
  const Index<2> extents{{{4, 2}}};
  Variables<tmpl::list<VariablesTestTags_detail::vector>> vars(
      extents.product());
  get<VariablesTestTags_detail::vector>(vars) =
      Vector{{{{1110000., 1120000., 1130000., 1140000., 1210000., 1220000.,
                1230000., 1240000.},
               {2110000., 2120000., 2130000., 2140000., 2210000., 2220000.,
                2230000., 2240000.},
               {3110000., 3120000., 3130000., 3140000., 3210000., 3220000.,
                3230000., 3240000.}}}};

  {
    const auto slice_extents = extents.slice_away(0);
    Variables<tmpl::list<VariablesTestTags_detail::vector>> slice(
        slice_extents.product(), 0.);
    get<VariablesTestTags_detail::vector>(slice) =
        Vector{{{{1100., 1200.}, {2100., 2200.}, {3100., 3200.}}}};
    add_slice_to_data(make_not_null(&vars), slice, extents, 0, 2);
  }
  {
    const auto slice_extents = extents.slice_away(1);
    Variables<tmpl::list<VariablesTestTags_detail::vector>> slice(
        slice_extents.product(), 0.);
    get<VariablesTestTags_detail::vector>(slice) = Vector{
        {{{11., 12., 13., 14.}, {21., 22., 23., 24.}, {31., 32., 33., 34.}}}};
    add_slice_to_data(make_not_null(&vars), slice, extents, 1, 1);
  }

  CHECK((Vector{{{{1110000., 1120000., 1131100., 1140000., 1210011., 1220012.,
                   1231213., 1240014.},
                  {2110000., 2120000., 2132100., 2140000., 2210021., 2220022.,
                   2232223., 2240024.},
                  {3110000., 3120000., 3133100., 3140000., 3210031., 3220032.,
                   3233233., 3240034.}}}}) ==
        get<VariablesTestTags_detail::vector>(vars));
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Variables.assign_to_default",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
  #ifdef SPECTRE_DEBUG
  Variables<tmpl::list<VariablesTestTags_detail::scalar>> vars;
  get<VariablesTestTags_detail::scalar>(vars) = Scalar<DataVector>{{{{0.}}}};
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, volume_vars has wrong number of grid points.
//  Expected 8, got 10]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Variables.add_slice_to_data.BadSize.volume",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Variables<tmpl::list<VariablesTestTags_detail::vector>> vars(10, 0.);
  const Variables<tmpl::list<VariablesTestTags_detail::vector>> slice(2, 0.);
  add_slice_to_data(make_not_null(&vars), slice, Index<2>{{{4, 2}}}, 0, 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, vars_on_slice has wrong number of grid points.
//  Expected 2, got 5]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Variables.add_slice_to_data.BadSize.slice",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Variables<tmpl::list<VariablesTestTags_detail::vector>> vars(8, 0.);
  const Variables<tmpl::list<VariablesTestTags_detail::vector>> slice(5, 0.);
  add_slice_to_data(make_not_null(&vars), slice, Index<2>{{{4, 2}}}, 0, 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
