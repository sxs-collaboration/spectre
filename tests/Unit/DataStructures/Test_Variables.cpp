// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "ErrorHandling/Error.hpp"  // IWYU pragma: keep
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"  // IWYU pragma: keep
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_include <boost/tuple/tuple.hpp>

// IWYU pragma: no_include "DataStructures/VariablesForwardDecl.hpp"

// IWYU pragma: no_forward_declare Variables

namespace VariablesTestTags_detail {
/// [simple_variables_tag]
template <typename VectorType>
struct tensor : db::SimpleTag {
  static std::string name() noexcept { return "tensor"; }
  using type = tnsr::I<VectorType, 3, Frame::Grid>;
};
/// [simple_variables_tag]
template <typename VectorType>
struct scalar : db::SimpleTag {
  static std::string name() noexcept { return "scalar"; }
  using type = Scalar<VectorType>;
};
template <typename VectorType>
struct scalar2 : db::SimpleTag {
  static std::string name() noexcept { return "scalar2"; }
  using type = Scalar<VectorType>;
};

/// [prefix_variables_tag]
template <class Tag>
struct Prefix0 : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "Prefix0"; }
};
/// [prefix_variables_tag]

template <class Tag>
struct Prefix1 : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "Prefix1"; }
};

template <class Tag>
struct Prefix2 : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "Prefix2"; }
};

template <class Tag>
struct Prefix3 : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "Prefix3"; }
};
}  // namespace VariablesTestTags_detail

static_assert(
    std::is_nothrow_move_constructible<Variables<
        tmpl::list<VariablesTestTags_detail::scalar<DataVector>,
                   VariablesTestTags_detail::tensor<DataVector>>>>::value,
    "Missing move semantics in Variables.");

namespace {
std::string repeat_string_with_commas(const std::string& str,
                                      const size_t repeats) noexcept {
  MakeString t;
  for (size_t i = 0; i < repeats - 1; ++i) {
    t << str << ",";
  }
  t << str;
  return t;
}

template <typename VectorType>
void test_variables_construction_and_access() noexcept {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{5, 20};

  const size_t number_of_grid_points = sdist(gen);
  // Test constructed and member function initialized Variables are identical
  const auto initial_fill_value = make_with_random_values<value_type>(
      make_not_null(&gen), make_not_null(&dist));

  using VariablesType =
      Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>,
                           VariablesTestTags_detail::scalar<VectorType>,
                           VariablesTestTags_detail::scalar2<VectorType>>>;

  VariablesType filled_variables{number_of_grid_points, initial_fill_value};

  VariablesType initialized_variables;
  initialized_variables.initialize(number_of_grid_points, initial_fill_value);
  CHECK(filled_variables == initialized_variables);

  CHECK(filled_variables.size() ==
        filled_variables.number_of_grid_points() *
            filled_variables.number_of_independent_components);
  CHECK(number_of_grid_points == filled_variables.number_of_grid_points());
  CHECK(number_of_grid_points * 5 == filled_variables.size());
  for (size_t i = 0; i < filled_variables.size(); ++i) {
    // clang-tidy: do not use pointer arithmetic
    CHECK(initial_fill_value == filled_variables.data()[i]);  // NOLINT
  }
  // Test that assigning to a tensor pointing into a Variables correctly sets
  // the Variables
  auto& tensor_in_filled_variables =
      get<VariablesTestTags_detail::tensor<VectorType>>(filled_variables);
  CHECK(VectorType{number_of_grid_points, initial_fill_value} ==
        get<0>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, initial_fill_value} ==
        get<1>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, initial_fill_value} ==
        get<2>(tensor_in_filled_variables));

  const auto value_for_reference_assignment =
      make_with_random_values<value_type>(make_not_null(&gen),
                                          make_not_null(&dist));

  tnsr::I<VectorType, 3, Frame::Grid> tensor_for_reference_assignment{
      number_of_grid_points, value_for_reference_assignment};
  tensor_in_filled_variables = tensor_for_reference_assignment;

  // clang-tidy: do not use pointer arithmetic
  for (size_t i = 0; i < number_of_grid_points * 3; ++i) {
    CHECK(value_for_reference_assignment ==
          filled_variables.data()[i]);  // NOLINT
  }
  CHECK(VectorType{number_of_grid_points, value_for_reference_assignment} ==
        get<0>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, value_for_reference_assignment} ==
        get<1>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, value_for_reference_assignment} ==
        get<2>(tensor_in_filled_variables));

  const auto value_for_tensor_constructor = make_with_random_values<value_type>(
      make_not_null(&gen), make_not_null(&dist));

  tensor_in_filled_variables = tnsr::I<VectorType, 3, Frame::Grid>{
      number_of_grid_points, value_for_tensor_constructor};

  // clang-tidy: do not use pointer arithmetic
  for (size_t i = 0; i < number_of_grid_points * 3; ++i) {
    CHECK(value_for_tensor_constructor ==
          filled_variables.data()[i]);  // NOLINT
  }
  CHECK(VectorType{number_of_grid_points, value_for_tensor_constructor} ==
        get<0>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, value_for_tensor_constructor} ==
        get<1>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, value_for_tensor_constructor} ==
        get<2>(tensor_in_filled_variables));

  // Test const vector in variables points to right values
  const auto& const_tensor_in_filled_variables =
      get<VariablesTestTags_detail::tensor<VectorType>>(filled_variables);
  CHECK(get<0>(const_tensor_in_filled_variables) ==
        VectorType{number_of_grid_points, value_for_tensor_constructor});
  CHECK(get<1>(const_tensor_in_filled_variables) ==
        VectorType{number_of_grid_points, value_for_tensor_constructor});
  CHECK(get<2>(const_tensor_in_filled_variables) ==
        VectorType{number_of_grid_points, value_for_tensor_constructor});

  // Test equivalence and inequivalence operators
  VariablesType another_filled_variables{number_of_grid_points,
                                         initial_fill_value};
  // handle case of same value rolled twice on the random number generators
  CHECK((filled_variables == another_filled_variables) ==
        (initial_fill_value == value_for_tensor_constructor));
  another_filled_variables = filled_variables;
  CHECK(another_filled_variables == filled_variables);

  // Test default constructed variables is allowed
  VariablesType default_variables;

  CHECK(default_variables.size() == 0);
  CHECK(default_variables.number_of_grid_points() == 0);
  default_variables = another_filled_variables;
  CHECK(another_filled_variables == default_variables);

  std::string expected_output_initial = repeat_string_with_commas(
      get_output(initial_fill_value), number_of_grid_points);

  std::string expected_output_tensor = repeat_string_with_commas(
      get_output(value_for_tensor_constructor), number_of_grid_points);

  // Test stream operator
  const std::string expected_output =
      "tensor:\n"
      "T(0)=(" +
      expected_output_tensor +
      ")\n"
      "T(1)=(" +
      expected_output_tensor +
      ")\n"
      "T(2)=(" +
      expected_output_tensor +
      ")\n\n"
      "scalar:\n"
      "T()=(" +
      expected_output_initial +
      ")\n\n"
      "scalar2:\n"
      "T()=(" +
      expected_output_initial + ")";
  CHECK(get_output(filled_variables) == expected_output);

  // Check self-assignment
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
#endif  // defined(__clang__) && __clang_major__ > 6
  filled_variables = filled_variables;  // NOLINT
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic pop
#endif  // defined(__clang__) && __clang_major__ > 6
  CHECK(filled_variables == another_filled_variables);

  CHECK(
      Tags::Variables<
          tmpl::list<VariablesTestTags_detail::tensor<VectorType>,
                     VariablesTestTags_detail::scalar<VectorType>,
                     VariablesTestTags_detail::scalar2<VectorType>>>::name() ==
      "Variables(tensor,scalar,scalar2)");
}

template <typename VectorType>
void test_variables_move() noexcept {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{5, 20};

  const size_t number_of_grid_points1 = sdist(gen);
  const size_t number_of_grid_points2 = sdist(gen);
  const std::array<value_type, 2> initial_fill_values =
      make_with_random_values<std::array<value_type, 2>>(make_not_null(&gen),
                                                         make_not_null(&dist));

  // Test moving with assignment operator
  Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>> move_to{
      number_of_grid_points1, initial_fill_values[0]};
  Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>> move_from{
      number_of_grid_points2, initial_fill_values[1]};

  Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>
      initializer_move_to = std::move(move_to);

  move_to = std::move(move_from);
  CHECK(move_to ==
        Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>{
            number_of_grid_points2, initial_fill_values[1]});
  CHECK(&get<VariablesTestTags_detail::tensor<VectorType>>(move_to)[0][0] ==
        move_to.data());

  // Intentionally testing self-move
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif  // defined(__clang__)
  move_to = std::move(move_to);
#ifdef __clang__
#pragma GCC diagnostic pop
#endif  // defined(__clang__)
  // clang-tidy: false positive 'move_to' used after it was moved
  CHECK(move_to ==  // NOLINT
        Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>{
            number_of_grid_points2, initial_fill_values[1]});
  CHECK(&get<VariablesTestTags_detail::tensor<VectorType>>(move_to)[0][0] ==
        move_to.data());
}

template <typename VectorType>
void test_variables_math() noexcept {
  using value_type = typename VectorType::value_type;
  using TestVariablesType =
      Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>,
                           VariablesTestTags_detail::scalar<VectorType>,
                           VariablesTestTags_detail::scalar2<VectorType>>>;

  MAKE_GENERATOR(gen);
  // keep distribution somewhat near 1.0 to avoid accumulation of errors in
  // cascaded math
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{1.0,
                                                                         10.0};
  UniformCustomDistribution<size_t> sdist{5, 20};

  const size_t num_points = sdist(gen);  // number of grid points
  const auto value_in_variables = make_with_random_values<value_type>(
      make_not_null(&gen), make_not_null(&dist));
  const std::array<value_type, 3> rand_vals =
      make_with_random_values<std::array<value_type, 3>>(make_not_null(&gen),
                                                         make_not_null(&dist));

  // Test math +, -, *, /
  const TestVariablesType vars{num_points, value_in_variables};
  TestVariablesType expected{num_points, rand_vals.at(0) * value_in_variables};
  CHECK_VARIABLES_APPROX(expected, rand_vals[0] * vars);
  expected = TestVariablesType{num_points, value_in_variables * rand_vals[0]};
  CHECK_VARIABLES_APPROX(expected, vars * rand_vals[0]);
  expected = TestVariablesType{num_points, value_in_variables / rand_vals[0]};
  CHECK_VARIABLES_APPROX(expected, vars / rand_vals[0]);

  expected = TestVariablesType{num_points, 4. * value_in_variables};
  CHECK_VARIABLES_APPROX(expected, vars + vars + vars + vars);
  expected = TestVariablesType{num_points, 3. * value_in_variables};
  CHECK_VARIABLES_APPROX(expected, vars + (vars + vars));
  // clang-tidy: both sides of overloaded operator are equivalent
  expected = TestVariablesType{num_points, -2. * value_in_variables};
  CHECK_VARIABLES_APPROX(expected, vars - vars - vars - vars);  // NOLINT
  expected = TestVariablesType{num_points, value_in_variables};
  CHECK_VARIABLES_APPROX(expected, vars - (vars - vars));  // NOLINT
  expected = TestVariablesType{num_points, -value_in_variables};
  CHECK_VARIABLES_APPROX(expected, vars - (vars + vars));
  expected = TestVariablesType{num_points, value_in_variables};
  CHECK_VARIABLES_APPROX(expected, vars - vars + vars);  // NOLINT
  expected = TestVariablesType{num_points, value_in_variables};
  CHECK_VARIABLES_APPROX(expected, vars + vars - vars);

  // Test math_assignment operators +=, -=, *=, /= with values
  TestVariablesType test_assignment(vars * rand_vals[0]);
  auto expected_val = value_in_variables * rand_vals[0];
  test_assignment += TestVariablesType{num_points, value_in_variables};
  expected_val += value_in_variables;
  expected = TestVariablesType{num_points, expected_val};
  CHECK_VARIABLES_APPROX(expected, test_assignment);
  test_assignment -= TestVariablesType{num_points, rand_vals[1]};
  expected_val -= rand_vals[1];
  expected = TestVariablesType{num_points, expected_val};
  CHECK_VARIABLES_APPROX(expected, test_assignment);
  test_assignment *= rand_vals[2];
  expected_val *= rand_vals[2];
  expected = TestVariablesType{num_points, expected_val};
  CHECK_VARIABLES_APPROX(expected, test_assignment);
  test_assignment /= rand_vals[0];
  expected_val /= rand_vals[0];
  expected = TestVariablesType{num_points, expected_val};
  CHECK_VARIABLES_APPROX(expected, test_assignment);

  test_assignment +=
      TestVariablesType{num_points, value_in_variables} * rand_vals[1];
  expected_val += value_in_variables * rand_vals[1];
  expected = TestVariablesType{num_points, expected_val};
  CHECK_VARIABLES_APPROX(expected, test_assignment);
  test_assignment -= TestVariablesType{num_points, rand_vals[2]} * rand_vals[0];
  expected_val -= rand_vals[2] * rand_vals[0];
  expected = TestVariablesType{num_points, expected_val};
  CHECK_VARIABLES_APPROX(expected, test_assignment);

  TestVariablesType test_assignment2{num_points};
  test_assignment2 = test_assignment * 1.0;
  CHECK(test_assignment2 == test_assignment);

  const auto check_components = [](const auto& variables,
                                   const VectorType& tensor) noexcept {
    tmpl::for_each<typename std::decay_t<decltype(variables)>::tags_list>(
        [&variables, &tensor ](auto tag) noexcept {
          using Tag = tmpl::type_from<decltype(tag)>;
          for (const auto& component : get<Tag>(variables)) {
            CHECK(component == tensor);
          }
        });
  };

  const VectorType test_vector = make_with_random_values<VectorType>(
      make_not_null(&gen), make_not_null(&dist), VectorType{4});

  // Test math assignment operators +=, -=, *=, /= with vectors
  TestVariablesType test_vector_math(4, rand_vals[0]);
  test_vector_math *= test_vector;
  VectorType vec_expect = rand_vals[0] * test_vector;
  check_components(test_vector_math, vec_expect);
  vec_expect *= test_vector;
  check_components(test_vector_math * test_vector, vec_expect);
  check_components(test_vector * test_vector_math, vec_expect);
  test_vector_math *= test_vector;
  check_components(test_vector_math, vec_expect);
  vec_expect /= test_vector;
  check_components(test_vector_math / test_vector, vec_expect);
  test_vector_math /= test_vector;
  check_components(test_vector_math, vec_expect);
}

template <typename VectorType>
void test_variables_prefix_semantics() noexcept {
  using TagList = tmpl::list<VariablesTestTags_detail::tensor<VectorType>,
                             VariablesTestTags_detail::scalar<VectorType>,
                             VariablesTestTags_detail::scalar2<VectorType>>;
  using VariablesType = Variables<TagList>;
  using PrefixVariablesType =
      Variables<db::wrap_tags_in<VariablesTestTags_detail::Prefix0, TagList>>;
  using value_type = typename VectorType::value_type;

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{5, 20};

  const size_t number_of_grid_points = sdist(gen);
  const std::array<value_type, 4> variables_vals =
      make_with_random_values<std::array<value_type, 4>>(make_not_null(&gen),
                                                         make_not_null(&dist));

  // Check move and copy from prefix variables to non-prefix variables via
  // constructor
  PrefixVariablesType prefix_vars_copy_construct_from{number_of_grid_points,
                                                      variables_vals[0]};
  PrefixVariablesType prefix_vars_move_construct_from{number_of_grid_points,
                                                      variables_vals[1]};
  VariablesType vars_to{prefix_vars_copy_construct_from};
  VariablesType move_constructed_vars{
      std::move(prefix_vars_move_construct_from)};

  VariablesType expected{number_of_grid_points, variables_vals[0]};
  CHECK_VARIABLES_APPROX(vars_to, expected);
  expected = VariablesType(number_of_grid_points, variables_vals[1]);
  CHECK_VARIABLES_APPROX(move_constructed_vars, expected);

  // Check move and copy from prefix variables to non-prefix variables via
  // assignment operator
  PrefixVariablesType prefix_vars_copy_assign_from{number_of_grid_points,
                                                   variables_vals[2]};
  PrefixVariablesType prefix_vars_move_assign_from{number_of_grid_points,
                                                   variables_vals[3]};
  vars_to = prefix_vars_copy_assign_from;
  CHECK_VARIABLES_APPROX(
      vars_to, VariablesType(number_of_grid_points, variables_vals[2]));
  vars_to = std::move(prefix_vars_move_assign_from);
  CHECK_VARIABLES_APPROX(
      vars_to, VariablesType(number_of_grid_points, variables_vals[3]));
}

template <typename VectorType>
void test_variables_prefix_math() noexcept {
  using TagList = tmpl::list<VariablesTestTags_detail::tensor<VectorType>,
                             VariablesTestTags_detail::scalar<VectorType>,
                             VariablesTestTags_detail::scalar2<VectorType>>;
  using Prefix0VariablesType =
      Variables<db::wrap_tags_in<VariablesTestTags_detail::Prefix0, TagList>>;
  using Prefix1VariablesType =
      Variables<db::wrap_tags_in<VariablesTestTags_detail::Prefix1, TagList>>;
  using Prefix2VariablesType =
      Variables<db::wrap_tags_in<VariablesTestTags_detail::Prefix2, TagList>>;
  using Prefix3VariablesType =
      Variables<db::wrap_tags_in<VariablesTestTags_detail::Prefix3, TagList>>;
  using value_type = typename VectorType::value_type;

  MAKE_GENERATOR(gen);
  // keep distribution somewhat near 1.0 to avoid accumulation of errors in
  // cascaded math
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{1.0,
                                                                         10.0};
  UniformCustomDistribution<size_t> sdist{5, 20};

  const size_t number_of_grid_points = sdist(gen);
  const auto value_in_variables = make_with_random_values<value_type>(
      make_not_null(&gen), make_not_null(&dist));
  const std::array<value_type, 3> random_vals =
      make_with_random_values<std::array<value_type, 3>>(make_not_null(&gen),
                                                         make_not_null(&dist));

  // Test arithmetic operators +, -, *, / for prefix Variables.

  // We wish to verify that operations among multiple Variables with distinct
  // prefixes give a Variables populated with the desired data. This is
  // important for multiplying, for instance a variables populated with time
  // derivatives with a variables populated with time intervals, which would
  // have the same types, but have distinct prefixes.
  Prefix0VariablesType prefix_vars0(number_of_grid_points, value_in_variables);
  Prefix1VariablesType prefix_vars1(number_of_grid_points, value_in_variables);
  Prefix2VariablesType prefix_vars2(number_of_grid_points, value_in_variables);
  Prefix3VariablesType prefix_vars3(number_of_grid_points, value_in_variables);

  Prefix3VariablesType expected{number_of_grid_points,
                                value_in_variables * random_vals[0]};
  CHECK_VARIABLES_APPROX(expected, random_vals[0] * prefix_vars0);
  expected = Prefix3VariablesType{number_of_grid_points,
                                  value_in_variables * random_vals[0]};
  CHECK_VARIABLES_APPROX(expected, prefix_vars0 * random_vals[0]);
  expected = Prefix3VariablesType{number_of_grid_points,
                                  value_in_variables / random_vals[0]};
  CHECK_VARIABLES_APPROX(expected, prefix_vars0 / random_vals[0]);
  expected =
      Prefix3VariablesType{number_of_grid_points, 4.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected, prefix_vars0 + prefix_vars1 + prefix_vars2 + prefix_vars3);
  expected =
      Prefix3VariablesType{number_of_grid_points, 4.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected, (prefix_vars0 + prefix_vars1) + (prefix_vars2 + prefix_vars3));
  expected =
      Prefix3VariablesType{number_of_grid_points, 3.0 * value_in_variables};

  CHECK_VARIABLES_APPROX(expected,
                         prefix_vars0 + (prefix_vars1 + prefix_vars2));

  // clang-tidy: both sides of overloaded operator are equivalent
  expected =
      Prefix3VariablesType{number_of_grid_points, -2.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected,
      prefix_vars0 - prefix_vars1 - prefix_vars2 - prefix_vars3);  // NOLINT
  expected =
      Prefix3VariablesType{number_of_grid_points, 2.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected,
      prefix_vars0 - (prefix_vars1 - prefix_vars2 - prefix_vars3));  // NOLINT
  expected =
      Prefix3VariablesType{number_of_grid_points, 0.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected,
      (prefix_vars0 - prefix_vars1) - (prefix_vars2 - prefix_vars3));  // NOLINT
  expected =
      Prefix3VariablesType{number_of_grid_points, 1.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected, prefix_vars0 - (prefix_vars1 - prefix_vars2));  // NOLINT
  expected =
      Prefix3VariablesType{number_of_grid_points, 1.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(expected,
                         prefix_vars0 - prefix_vars1 + prefix_vars2);  // NOLINT
  expected =
      Prefix3VariablesType{number_of_grid_points, 1.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(expected, prefix_vars0 + prefix_vars1 - prefix_vars2);

  // Test assignment arithmetic operators +=, -=, *=, /= with values
  Prefix0VariablesType test_assignment(prefix_vars0 * 1.0);
  test_assignment +=
      Prefix1VariablesType{number_of_grid_points, value_in_variables};
  auto expected_val = 2.0 * value_in_variables;
  Prefix0VariablesType expected_prefix0{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);
  test_assignment -=
      Prefix1VariablesType{number_of_grid_points, random_vals[0]};
  expected_val -= random_vals[0];
  expected_prefix0 = Prefix0VariablesType{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);
  test_assignment *= random_vals[1];
  expected_val *= random_vals[1];
  expected_prefix0 = Prefix0VariablesType{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);
  test_assignment /= random_vals[2];
  expected_val /= random_vals[2];
  expected_prefix0 = Prefix0VariablesType{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);

  test_assignment +=
      Prefix1VariablesType{number_of_grid_points, random_vals[0]} *
      random_vals[1];
  expected_val += random_vals[0] * random_vals[1];
  expected_prefix0 = Prefix0VariablesType{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);
  test_assignment -=
      Prefix1VariablesType{number_of_grid_points, random_vals[0]} *
      random_vals[2];
  expected_val -= random_vals[0] * random_vals[2];
  expected_prefix0 = Prefix0VariablesType{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);
}

template <typename VectorType>
void test_variables_serialization() noexcept {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{5, 20};

  const size_t number_of_grid_points = sdist(gen);
  const std::array<value_type, 2> values_in_variables =
      make_with_random_values<std::array<value_type, 2>>(make_not_null(&gen),
                                                         make_not_null(&dist));

  // Test serialization of a Variables and a tuple of Variables
  Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>
      test_variables(number_of_grid_points, values_in_variables[0]);
  test_serialization(test_variables);
  auto tuple_of_test_variables = std::make_tuple(
      Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>{
          number_of_grid_points, values_in_variables[1]});
  test_serialization(tuple_of_test_variables);
}

template <typename VectorType>
void test_variables_assign_subset() noexcept {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{5, 20};

  const size_t number_of_grid_points = sdist(gen);
  const std::array<value_type, 4> values_in_variables =
      make_with_random_values<std::array<value_type, 4>>(make_not_null(&gen),
                                                         make_not_null(&dist));

  // Test assigning a single tag to a Variables with two tags, either from a
  // Variables or a TaggedTuple
  const auto test_assign_to_vars_with_two_tags =
      [&number_of_grid_points, &values_in_variables ](
          const auto& vars_subset0, const auto& vars_subset1,
          const value_type& vars_subset0_val,
          const value_type& vars_subset1_val) noexcept {
    Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>,
                         VariablesTestTags_detail::scalar<VectorType>>>
        vars_set{number_of_grid_points, values_in_variables[0]};
    CHECK(get<VariablesTestTags_detail::tensor<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::tensor<VectorType>>(
              number_of_grid_points, values_in_variables[0]));
    CHECK(get<VariablesTestTags_detail::scalar<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::scalar<VectorType>>(
              number_of_grid_points, values_in_variables[0]));
    vars_set.assign_subset(vars_subset0);
    CHECK(get<VariablesTestTags_detail::tensor<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::tensor<VectorType>>(
              number_of_grid_points, vars_subset0_val));
    CHECK(get<VariablesTestTags_detail::scalar<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::scalar<VectorType>>(
              number_of_grid_points, values_in_variables[0]));
    vars_set.assign_subset(vars_subset1);
    CHECK(get<VariablesTestTags_detail::tensor<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::tensor<VectorType>>(
              number_of_grid_points, vars_subset0_val));
    CHECK(get<VariablesTestTags_detail::scalar<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::scalar<VectorType>>(
              number_of_grid_points, vars_subset1_val));
  };

  test_assign_to_vars_with_two_tags(
      Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>(
          number_of_grid_points, values_in_variables[1]),
      Variables<tmpl::list<VariablesTestTags_detail::scalar<VectorType>>>(
          number_of_grid_points, values_in_variables[2]),
      values_in_variables[1], values_in_variables[2]);
  test_assign_to_vars_with_two_tags(
      tuples::TaggedTuple<VariablesTestTags_detail::tensor<VectorType>>(
          typename VariablesTestTags_detail::tensor<VectorType>::type{
              number_of_grid_points, values_in_variables[1]}),
      tuples::TaggedTuple<VariablesTestTags_detail::scalar<VectorType>>(
          typename VariablesTestTags_detail::scalar<VectorType>::type{
              number_of_grid_points, values_in_variables[2]}),
      values_in_variables[1], values_in_variables[2]);

  // Test assigning a single tag to a Variables with three tags, either from a
  // Variables or a TaggedTuple
  const auto test_assign_to_vars_with_three_tags =
      [&number_of_grid_points, &values_in_variables ](
          const auto& vars_subset0, const auto& vars_subset1,
          const value_type& vars_subset0_val,
          const value_type& vars_subset1_val) noexcept {
    Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>,
                         VariablesTestTags_detail::scalar<VectorType>,
                         VariablesTestTags_detail::scalar2<VectorType>>>
        vars_set(number_of_grid_points, values_in_variables[0]);
    CHECK(get<VariablesTestTags_detail::tensor<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::tensor<VectorType>>(
              number_of_grid_points, values_in_variables[0]));
    CHECK(get<VariablesTestTags_detail::scalar<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::scalar<VectorType>>(
              number_of_grid_points, values_in_variables[0]));
    CHECK(get<VariablesTestTags_detail::scalar2<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::scalar2<VectorType>>(
              number_of_grid_points, values_in_variables[0]));
    vars_set.assign_subset(vars_subset0);
    CHECK(get<VariablesTestTags_detail::tensor<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::tensor<VectorType>>(
              number_of_grid_points, vars_subset0_val));
    CHECK(get<VariablesTestTags_detail::scalar<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::scalar<VectorType>>(
              number_of_grid_points, values_in_variables[0]));
    CHECK(get<VariablesTestTags_detail::scalar2<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::scalar2<VectorType>>(
              number_of_grid_points, values_in_variables[0]));
    vars_set.assign_subset(vars_subset1);
    CHECK(get<VariablesTestTags_detail::tensor<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::tensor<VectorType>>(
              number_of_grid_points, vars_subset0_val));
    CHECK(get<VariablesTestTags_detail::scalar<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::scalar<VectorType>>(
              number_of_grid_points, vars_subset1_val));
    CHECK(get<VariablesTestTags_detail::scalar2<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::scalar2<VectorType>>(
              number_of_grid_points, values_in_variables[0]));
  };

  test_assign_to_vars_with_three_tags(
      Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>(
          number_of_grid_points, values_in_variables[1]),
      Variables<tmpl::list<VariablesTestTags_detail::scalar<VectorType>>>(
          number_of_grid_points, values_in_variables[2]),
      values_in_variables[1], values_in_variables[2]);
  test_assign_to_vars_with_three_tags(
      tuples::TaggedTuple<VariablesTestTags_detail::tensor<VectorType>>(
          typename VariablesTestTags_detail::tensor<VectorType>::type{
              number_of_grid_points, values_in_variables[1]}),
      tuples::TaggedTuple<VariablesTestTags_detail::scalar<VectorType>>(
          typename VariablesTestTags_detail::scalar<VectorType>::type{
              number_of_grid_points, values_in_variables[2]}),
      values_in_variables[1], values_in_variables[2]);

  // Test assignment to a Variables with a single tag either from a Variables or
  // a TaggedTuple
  const auto test_assign_to_vars_with_one_tag = [
    &number_of_grid_points, &values_in_variables
  ](const auto& vars_subset0, const value_type& vars_subset0_val) noexcept {
    Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>
        vars_set(number_of_grid_points, values_in_variables[0]);
    CHECK(get<VariablesTestTags_detail::tensor<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::tensor<VectorType>>(
              number_of_grid_points, values_in_variables[0]));
    vars_set.assign_subset(vars_subset0);
    CHECK(get<VariablesTestTags_detail::tensor<VectorType>>(vars_set) ==
          db::item_type<VariablesTestTags_detail::tensor<VectorType>>(
              number_of_grid_points, vars_subset0_val));
  };

  test_assign_to_vars_with_one_tag(
      Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>(
          number_of_grid_points, values_in_variables[1]),
      values_in_variables[1]);
  test_assign_to_vars_with_one_tag(
      tuples::TaggedTuple<VariablesTestTags_detail::tensor<VectorType>>(
          typename VariablesTestTags_detail::tensor<VectorType>::type{
              number_of_grid_points, values_in_variables[1]}),
      values_in_variables[1]);
}

template <typename VectorType>
void test_variables_extract_subset() noexcept {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{5, 20};

  const size_t number_of_grid_points = sdist(gen);

  const auto vars_subset0 = make_with_random_values<
      Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>>(
      make_not_null(&gen), make_not_null(&dist),
      VectorType{number_of_grid_points});
  const auto vars_subset1 = make_with_random_values<
      Variables<tmpl::list<VariablesTestTags_detail::scalar<VectorType>>>>(
      make_not_null(&gen), make_not_null(&dist),
      VectorType{number_of_grid_points});

  Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>,
                       VariablesTestTags_detail::scalar<VectorType>>>
      vars(number_of_grid_points);
  get<VariablesTestTags_detail::tensor<VectorType>>(vars) =
      get<VariablesTestTags_detail::tensor<VectorType>>(vars_subset0);
  get<VariablesTestTags_detail::scalar<VectorType>>(vars) =
      get<VariablesTestTags_detail::scalar<VectorType>>(vars_subset1);
  CHECK(vars.template extract_subset<
            tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>() ==
        vars_subset0);
  CHECK(vars.template extract_subset<
            tmpl::list<VariablesTestTags_detail::scalar<VectorType>>>() ==
        vars_subset1);
  CHECK(vars.template extract_subset<
            tmpl::list<VariablesTestTags_detail::tensor<VectorType>,
                       VariablesTestTags_detail::scalar<VectorType>>>() ==
        vars);
}

template <typename VectorType>
void test_variables_slice() noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{5, 10};

  const size_t x_extents = sdist(gen);
  const size_t y_extents = sdist(gen);
  const size_t z_extents = sdist(gen);
  Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>> vars{
      x_extents * y_extents * z_extents};
  const size_t tensor_size =
      VariablesTestTags_detail::tensor<VectorType>::type::size();
  Index<3> extents(x_extents, y_extents, z_extents);

  // Test data_on_slice function by using a predictable data set where each
  // entry is assigned a value equal to its index
  for (size_t s = 0; s < vars.size(); ++s) {
    // clang-tidy: do not use pointer arithmetic
    vars.data()[s] = s;  // NOLINT
  }
  Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>>
      expected_vars_sliced_in_x(y_extents * z_extents, 0.),
      expected_vars_sliced_in_y(x_extents * z_extents, 0.),
      expected_vars_sliced_in_z(x_extents * y_extents, 0.);
  const size_t x_offset = sdist(gen) % x_extents;
  const size_t y_offset = sdist(gen) % y_extents;
  const size_t z_offset = sdist(gen) % z_extents;

  for (size_t s = 0; s < expected_vars_sliced_in_x.size(); ++s) {
    // clang-tidy: do not use pointer arithmetic
    expected_vars_sliced_in_x.data()[s] = x_offset + s * x_extents;  // NOLINT
  }
  for (size_t i = 0; i < tensor_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t z = 0; z < z_extents; ++z) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_y
            .data()[x + x_extents * (z + z_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y_offset + z * y_extents);
      }
    }
  }
  for (size_t i = 0; i < tensor_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t y = 0; y < y_extents; ++y) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_z
            .data()[x + x_extents * (y + y_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y + y_extents * z_offset);
      }
    }
  }

  CHECK(data_on_slice(get<VariablesTestTags_detail::tensor<VectorType>>(vars),
                      extents, 0, x_offset) ==
        get<VariablesTestTags_detail::tensor<VectorType>>(
            expected_vars_sliced_in_x));
  CHECK(data_on_slice(get<VariablesTestTags_detail::tensor<VectorType>>(vars),
                      extents, 1, y_offset) ==
        get<VariablesTestTags_detail::tensor<VectorType>>(
            expected_vars_sliced_in_y));
  CHECK(data_on_slice(get<VariablesTestTags_detail::tensor<VectorType>>(vars),
                      extents, 2, z_offset) ==
        get<VariablesTestTags_detail::tensor<VectorType>>(
            expected_vars_sliced_in_z));

  CHECK(data_on_slice(vars, extents, 0, x_offset) == expected_vars_sliced_in_x);
  CHECK(data_on_slice(vars, extents, 1, y_offset) == expected_vars_sliced_in_y);
  CHECK(data_on_slice(vars, extents, 2, z_offset) == expected_vars_sliced_in_z);

  CHECK(data_on_slice<VariablesTestTags_detail::tensor<VectorType>>(
            extents, 0, x_offset,
            get<VariablesTestTags_detail::tensor<VectorType>>(vars)) ==
        expected_vars_sliced_in_x);
  CHECK(data_on_slice<VariablesTestTags_detail::tensor<VectorType>>(
            extents, 1, y_offset,
            get<VariablesTestTags_detail::tensor<VectorType>>(vars)) ==
        expected_vars_sliced_in_y);
  CHECK(data_on_slice<VariablesTestTags_detail::tensor<VectorType>>(
            extents, 2, z_offset,
            get<VariablesTestTags_detail::tensor<VectorType>>(vars)) ==
        expected_vars_sliced_in_z);
}

template <typename VectorType>
void test_variables_add_slice_to_data() noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<
      tt::get_fundamental_type_t<typename VectorType::value_type>>
      dist{-100.0, 100.0};

  // Test adding two slices on different 'axes' to a Variables
  std::array<VectorType, 3> orig_vals;
  std::fill(orig_vals.begin(), orig_vals.end(), VectorType{8});
  fill_with_random_values(make_not_null(&orig_vals), make_not_null(&gen),
                          make_not_null(&dist));

  std::array<VectorType, 3> slice0_vals;
  std::fill(slice0_vals.begin(), slice0_vals.end(), VectorType{4});
  fill_with_random_values(make_not_null(&slice0_vals), make_not_null(&gen),
                          make_not_null(&dist));

  std::array<VectorType, 3> slice1_vals;
  std::fill(slice1_vals.begin(), slice1_vals.end(), VectorType{2});
  fill_with_random_values(make_not_null(&slice1_vals), make_not_null(&gen),
                          make_not_null(&dist));

  using Tensor = typename VariablesTestTags_detail::tensor<VectorType>::type;
  const Index<2> extents{{{4, 2}}};
  Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>> vars(
      extents.product());
  get<VariablesTestTags_detail::tensor<VectorType>>(vars) =
      Tensor{{{orig_vals[0], orig_vals[1], orig_vals[2]}}};
  {
    const auto slice_extents = extents.slice_away(0);
    Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>> slice(
        slice_extents.product(), 0.);
    get<VariablesTestTags_detail::tensor<VectorType>>(slice) =
        Tensor{{{slice1_vals[0], slice1_vals[1], slice1_vals[2]}}};
    add_slice_to_data(make_not_null(&vars), slice, extents, 0, 2);
  }

  {
    const auto slice_extents = extents.slice_away(1);
    Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>>> slice(
        slice_extents.product(), 0.);
    get<VariablesTestTags_detail::tensor<VectorType>>(slice) =
        Tensor{{{slice0_vals[0], slice0_vals[1], slice0_vals[2]}}};
    add_slice_to_data(make_not_null(&vars), slice, extents, 1, 1);
  }

  // The slice0_vals should have been added to the second half of each of the
  // three vectors in the tensor. The slice1_vals should have been added to
  // entries 2 and 6 in each vector.
  // clang-format off
  const Tensor expected{
      {{{orig_vals[0].at(0),
         orig_vals[0].at(1),
         orig_vals[0].at(2) + slice1_vals[0].at(0),
         orig_vals[0].at(3),
         orig_vals[0].at(4) + slice0_vals[0].at(0),
         orig_vals[0].at(5) + slice0_vals[0].at(1),
         orig_vals[0].at(6) + slice0_vals[0].at(2) + slice1_vals[0].at(1),
         orig_vals[0].at(7) + slice0_vals[0].at(3)},
        {orig_vals[1].at(0),
         orig_vals[1].at(1),
         orig_vals[1].at(2) + slice1_vals[1].at(0),
         orig_vals[1].at(3),
         orig_vals[1].at(4) + slice0_vals[1].at(0),
         orig_vals[1].at(5) + slice0_vals[1].at(1),
         orig_vals[1].at(6) + slice0_vals[1].at(2) + slice1_vals[1].at(1),
         orig_vals[1].at(7) + slice0_vals[1].at(3)},
        {orig_vals[2].at(0),
         orig_vals[2].at(1),
         orig_vals[2].at(2) + slice1_vals[2].at(0),
         orig_vals[2].at(3),
         orig_vals[2].at(4) + slice0_vals[2].at(0),
         orig_vals[2].at(5) + slice0_vals[2].at(1),
         orig_vals[2].at(6) + slice0_vals[2].at(2) + slice1_vals[2].at(1),
         orig_vals[2].at(7) + slice0_vals[2].at(3)}}}};
  // clang-format on

  CHECK_ITERABLE_APPROX(
      expected, get<VariablesTestTags_detail::tensor<VectorType>>(vars));
}

template <typename VectorType>
void test_variables_from_tagged_tuple() noexcept {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{5, 20};

  const size_t number_of_grid_points = sdist(gen);
  tuples::TaggedTuple<VariablesTestTags_detail::tensor<VectorType>,
                      VariablesTestTags_detail::scalar<VectorType>>
      source;
  get<VariablesTestTags_detail::tensor<VectorType>>(source) =
      make_with_random_values<
          typename VariablesTestTags_detail::tensor<VectorType>::type>(
          make_not_null(&gen), make_not_null(&dist),
          VectorType{number_of_grid_points});
  get<VariablesTestTags_detail::scalar<VectorType>>(source) =
      make_with_random_values<
          typename VariablesTestTags_detail::scalar<VectorType>::type>(
          make_not_null(&gen), make_not_null(&dist),
          VectorType{number_of_grid_points});

  Variables<tmpl::list<VariablesTestTags_detail::tensor<VectorType>,
                       VariablesTestTags_detail::scalar<VectorType>>>
      assigned(number_of_grid_points);
  assigned.assign_subset(source);
  const auto created = variables_from_tagged_tuple(source);
  CHECK(assigned == created);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Variables", "[DataStructures][Unit]") {
  SECTION("Test Variables construction, access, and assignment") {
    test_variables_construction_and_access<ComplexDataVector>();
    test_variables_construction_and_access<ComplexModalVector>();
    test_variables_construction_and_access<DataVector>();
    test_variables_construction_and_access<ModalVector>();
  }
  SECTION("Test Variables move operations") {
    test_variables_move<ComplexDataVector>();
    test_variables_move<ComplexModalVector>();
    test_variables_move<DataVector>();
    test_variables_move<ModalVector>();
  }
  SECTION("Test Variables arithmetic operations") {
    test_variables_math<DataVector>();
    test_variables_math<ComplexDataVector>();
    // tests for ModalVector and ComplexModalVector omitted due to limited
    // arithmetic operation support for ModalVectors
  }
  SECTION("Test Prefix Variables move and copy semantics") {
    test_variables_prefix_semantics<ComplexDataVector>();
    test_variables_prefix_semantics<ComplexModalVector>();
    test_variables_prefix_semantics<DataVector>();
    test_variables_prefix_semantics<ModalVector>();
  }
  SECTION("Test Prefix Variables arithmetic operations") {
    test_variables_prefix_math<ComplexDataVector>();
    test_variables_prefix_math<ComplexModalVector>();
    test_variables_prefix_math<DataVector>();
    test_variables_prefix_math<ModalVector>();
  }
  SECTION("Test Variables serialization") {
    test_variables_serialization<ComplexDataVector>();
    test_variables_serialization<ComplexModalVector>();
    test_variables_serialization<DataVector>();
    test_variables_serialization<ModalVector>();
  }
  SECTION("Test Variables assign subset") {
    test_variables_assign_subset<ComplexDataVector>();
    test_variables_assign_subset<ComplexModalVector>();
    test_variables_assign_subset<DataVector>();
    test_variables_assign_subset<ModalVector>();
  }
  SECTION("Test Variables extract subset") {
    test_variables_extract_subset<ComplexDataVector>();
    test_variables_extract_subset<ComplexModalVector>();
    test_variables_extract_subset<DataVector>();
    test_variables_extract_subset<ModalVector>();
  }
  SECTION("Test Variables slice utilities") {
    test_variables_slice<ComplexDataVector>();
    test_variables_slice<ComplexModalVector>();
    test_variables_slice<DataVector>();
    test_variables_slice<ModalVector>();
  }
  SECTION("Test adding slice values to Variables") {
    test_variables_add_slice_to_data<ComplexDataVector>();
    test_variables_add_slice_to_data<ComplexModalVector>();
    test_variables_add_slice_to_data<DataVector>();
    test_variables_add_slice_to_data<ModalVector>();
  }
  SECTION("Test variables_from_tagged_tuple") {
    // The commented functions require a fix to issue #1420.
    // test_variables_from_tagged_tuple<ComplexDataVector>();
    // test_variables_from_tagged_tuple<ComplexModalVector>();
    test_variables_from_tagged_tuple<DataVector>();
    // test_variables_from_tagged_tuple<ModalVector>();
  }
}
}  // namespace

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.DataStructures.Variables.BadCopy",
                               "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Variables<tmpl::list<VariablesTestTags_detail::tensor<DataVector>,
                       VariablesTestTags_detail::scalar<DataVector>,
                       VariablesTestTags_detail::scalar2<DataVector>>>
      vars(1, -3.0);
  auto& tensor_in_vars =
      get<VariablesTestTags_detail::tensor<DataVector>>(vars);
  tensor_in_vars = tnsr::I<DataVector, 3, Frame::Grid>{10_st, -4.0};
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

    // clang-format off
// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Variables.assign_to_default",
    "[DataStructures][Unit]") {
  // clang-format on
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Variables<tmpl::list<VariablesTestTags_detail::scalar<DataVector>>> vars;
  get<VariablesTestTags_detail::scalar<DataVector>>(vars) =
      Scalar<DataVector>{{{{0.}}}};
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
  Variables<tmpl::list<VariablesTestTags_detail::tensor<DataVector>>> vars(10,
                                                                           0.);
  const Variables<tmpl::list<VariablesTestTags_detail::tensor<DataVector>>>
      slice(2, 0.);
  add_slice_to_data(make_not_null(&vars), slice, Index<2>{{{4, 2}}}, 0, 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

    // clang-format off
// [[OutputRegex, vars_on_slice has wrong number of grid points.
//  Expected 2, got 5]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Variables.add_slice_to_data.BadSize.slice",
    "[DataStructures][Unit]") {
  // clang-format on
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Variables<tmpl::list<VariablesTestTags_detail::tensor<DataVector>>> vars(8,
                                                                           0.);
  const Variables<tmpl::list<VariablesTestTags_detail::tensor<DataVector>>>
      slice(5, 0.);
  add_slice_to_data(make_not_null(&vars), slice, Index<2>{{{4, 2}}}, 0, 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
