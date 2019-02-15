// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>  // IWYU pragma: keep
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/VectorImpl.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/DereferenceWrapper.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/TestingFramework.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace TestHelpers {
namespace VectorImpl {

/// \ingroup TestingFrameworkGroup
/// \brief test construction and assignment of a `VectorType` with a `ValueType`
template <typename VectorType, typename ValueType>
void vector_test_construct_and_assign(
    tt::get_fundamental_type_t<ValueType> low =
        tt::get_fundamental_type_t<ValueType>{-100.0},
    tt::get_fundamental_type_t<ValueType> high =
        tt::get_fundamental_type_t<ValueType>{100.0}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<ValueType>> dist{low,
                                                                        high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  const size_t size = sdist(gen);

  const VectorType size_constructed{size};
  CHECK(size_constructed.size() == size);
  const auto generated_value1 = make_with_random_values<ValueType>(
      make_not_null(&gen), make_not_null(&dist));

  const VectorType value_size_constructed{size, generated_value1};
  CHECK(value_size_constructed.size() == size);
  alg::for_each(value_size_constructed, [
    generated_value1, &value_size_constructed
  ](typename VectorType::value_type element) noexcept {
    CAPTURE_PRECISE(value_size_constructed);
    CHECK(element == generated_value1);
  });

  // random generation must use `make_with_random_values`, because stored value
  // in vector type might be a non-fundamental type.
  const auto generated_value2 = make_with_random_values<ValueType>(
      make_not_null(&gen), make_not_null(&dist));
  const auto generated_value3 = make_with_random_values<ValueType>(
      make_not_null(&gen), make_not_null(&dist));

  VectorType initializer_list_constructed{
      {static_cast<typename VectorType::value_type>(generated_value2),
       static_cast<typename VectorType::value_type>(generated_value3)}};
  CHECK(initializer_list_constructed.size() == 2);
  CHECK(initializer_list_constructed.is_owning());
  CHECK(gsl::at(initializer_list_constructed, 0) == generated_value2);
  CHECK(gsl::at(initializer_list_constructed, 1) == generated_value3);

  typename VectorType::value_type raw_ptr[2] = {generated_value2,
                                                generated_value3};
  const VectorType pointer_size_constructed{
      static_cast<typename VectorType::value_type*>(raw_ptr), 2};
  CHECK(initializer_list_constructed == pointer_size_constructed);
  CHECK_FALSE(initializer_list_constructed != pointer_size_constructed);

  test_copy_semantics(initializer_list_constructed);
  auto initializer_list_constructed_copy = initializer_list_constructed;
  CHECK(initializer_list_constructed_copy.is_owning());
  CHECK(initializer_list_constructed_copy == pointer_size_constructed);
  test_move_semantics(std::move(initializer_list_constructed),
                      initializer_list_constructed_copy);

  VectorType move_assignment_initialized;
  move_assignment_initialized = std::move(initializer_list_constructed_copy);
  CHECK(move_assignment_initialized.is_owning());

  const VectorType move_constructed{std::move(move_assignment_initialized)};
  CHECK(move_constructed.is_owning());
  CHECK(move_constructed == pointer_size_constructed);

  // clang-tidy has performance complaints, and we're checking functionality
  const VectorType copy_constructed{move_constructed};  // NOLINT
  CHECK(copy_constructed.is_owning());
  CHECK(copy_constructed == pointer_size_constructed);
}

/// \ingroup TestingFrameworkGroup
/// \brief test the serialization of a `VectorType` constructed with a
/// `ValueType`
template <typename VectorType, typename ValueType>
void vector_test_serialize(tt::get_fundamental_type_t<ValueType> low =
                               tt::get_fundamental_type_t<ValueType>{-100.0},
                           tt::get_fundamental_type_t<ValueType> high =
                               tt::get_fundamental_type_t<ValueType>{
                                   100.0}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<ValueType>> dist{low,
                                                                        high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  const size_t size = sdist(gen);
  VectorType vector_test{size}, vector_control{size};
  VectorType vector_ref;
  const auto start_value = make_with_random_values<ValueType>(
      make_not_null(&gen), make_not_null(&dist));
  const auto value_difference = make_with_random_values<ValueType>(
      make_not_null(&gen), make_not_null(&dist));
  // generate_series is used to generate a pair of equivalent, but independently
  // constructed, data sets to fill the vectors with.
  ValueType current_value = start_value;
  const auto generate_series = [&current_value, value_difference ]() noexcept {
    return current_value += value_difference;
  };
  std::generate(vector_test.begin(), vector_test.end(), generate_series);
  current_value = start_value;
  std::generate(vector_control.begin(), vector_control.end(), generate_series);
  // checks the vectors have been constructed as expected
  CHECK(vector_control == vector_test);
  CHECK(vector_test.is_owning());
  CHECK(vector_control.is_owning());
  const VectorType serialized_vector_test =
      serialize_and_deserialize(vector_test);
  // check that the vector is unaltered by serialize_and_deserialize
  CHECK(vector_control == vector_test);
  CHECK(serialized_vector_test == vector_control);
  CHECK(serialized_vector_test.is_owning());
  CHECK(serialized_vector_test.data() != vector_test.data());
  CHECK(vector_test.is_owning());
  // checks serialization for reference
  vector_ref.set_data_ref(&vector_test);
  CHECK(vector_test.is_owning());
  CHECK_FALSE(vector_ref.is_owning());
  CHECK(vector_ref == vector_test);
  const VectorType serialized_vector_ref =
      serialize_and_deserialize(vector_ref);
  CHECK(vector_test.is_owning());
  CHECK(vector_test == vector_control);
  CHECK(vector_ref == vector_test);
  CHECK(serialized_vector_ref == vector_test);
  CHECK(serialized_vector_ref.is_owning());
  CHECK(serialized_vector_ref.data() != vector_ref.data());
  CHECK_FALSE(vector_ref.is_owning());
}

/// \ingroup TestingFrameworkGroup
/// \brief test the construction and move of a reference `VectorType`
/// constructed with a `ValueType`
template <typename VectorType, typename ValueType>
void vector_test_ref(tt::get_fundamental_type_t<ValueType> low =
                         tt::get_fundamental_type_t<ValueType>{-100.0},
                     tt::get_fundamental_type_t<ValueType> high =
                         tt::get_fundamental_type_t<ValueType>{
                             100.0}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<ValueType>> dist{low,
                                                                        high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  const size_t size = sdist(gen);
  VectorType original_vector = make_with_random_values<VectorType>(
      make_not_null(&gen), make_not_null(&dist), VectorType{size});

  {
    INFO("Check construction, copy, move, and ownership of reference vectors")
    VectorType ref_vector;
    ref_vector.set_data_ref(&original_vector);
    CHECK_FALSE(ref_vector.is_owning());
    CHECK(original_vector.is_owning());
    CHECK(ref_vector.data() == original_vector.data());

    const VectorType data_check{original_vector};
    CHECK(ref_vector.size() == size);
    CHECK(ref_vector == data_check);
    test_copy_semantics(ref_vector);

    VectorType ref_vector_copy;
    ref_vector_copy.set_data_ref(&ref_vector);
    test_move_semantics(std::move(ref_vector), ref_vector_copy);
    VectorType move_assignment_initialized;
    move_assignment_initialized = std::move(ref_vector_copy);
    CHECK(not move_assignment_initialized.is_owning());
    const VectorType move_constructed{std::move(move_assignment_initialized)};
    CHECK(not move_constructed.is_owning());
  }
  {
    INFO("Check move acts appropriately on both source and target refs")
    VectorType ref_original_vector;
    ref_original_vector.set_data_ref(&original_vector);
    VectorType generated_vector = make_with_random_values<VectorType>(
        make_not_null(&gen), make_not_null(&dist), VectorType{size});
    const VectorType generated_vector_copy = generated_vector;
    ref_original_vector = std::move(generated_vector);
    // clang-tidy : Intentionally testing use after move
    CHECK(original_vector != generated_vector);  // NOLINT
    CHECK(original_vector == generated_vector_copy);
// Intentionally testing self-move
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif  // defined(__clang__)
    ref_original_vector = std::move(ref_original_vector);
#ifdef __clang__
#pragma GCC diagnostic pop
#endif  // defined(__clang__)
    CHECK(original_vector == generated_vector_copy);
    // clang-tidy: false positive, used after it was moved
    const VectorType data_check_vector = ref_original_vector;  // NOLINT
    CHECK(data_check_vector == generated_vector_copy);
  }
  {
    INFO("Check math affects both data vectors which share a ref")
    const auto generated_value1 = make_with_random_values<ValueType>(
        make_not_null(&gen), make_not_null(&dist));
    const auto generated_value2 = make_with_random_values<ValueType>(
        make_not_null(&gen), make_not_null(&dist));
    const auto sum_generated_values = generated_value1 + generated_value2;
    VectorType sharing_vector{size, generated_value1};
    VectorType owning_vector{size, generated_value2};
    sharing_vector.set_data_ref(&owning_vector);
    sharing_vector = sharing_vector + generated_value1;
    CHECK_ITERABLE_APPROX(owning_vector,
                          (VectorType{size, sum_generated_values}));
    CHECK_ITERABLE_APPROX(sharing_vector,
                          (VectorType{size, sum_generated_values}));
  }
}

enum RefSizeErrorTestKind { Copy, ExpressionAssign, Move };

/// \ingroup TestingFrameworkGroup
/// \brief Test that assigning to a non-owning `VectorType` of the wrong size
/// appropriately generates an error.
///
/// \details a calling function should be an `ASSERTION_TEST()` and check for
/// the string "Must copy into same size".
/// Three types of tests are provided and one must be provided as the first
/// function argument:
/// - `RefSizeErrorTestKind::Copy`: Checks that copy-assigning to a non-owning
/// `VectorType` from a `VectorType` with the wrong size generates an error.
/// - `RefSizeErrorTestKind::ExpressionAssign`: Checks that assigning to a
/// non-owning `VectorType` from an expression with alias `ResultType` of
/// `VectorType` with the wrong size generates an error
/// - `RefSizeErrorTestKind::Move`: Checks that move-assigning to a non-owning
/// `VectorType` from a `VectorType` with the wrong size generates an error.
template <typename VectorType,
          typename ValueType = typename VectorType::ElementType>
void vector_ref_test_size_error(
    RefSizeErrorTestKind test_kind,
    tt::get_fundamental_type_t<ValueType> low =
        tt::get_fundamental_type_t<ValueType>{-100.0},
    tt::get_fundamental_type_t<ValueType> high =
        tt::get_fundamental_type_t<ValueType>{100.0}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<ValueType>> dist{low,
                                                                        high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  const size_t size = sdist(gen);
  VectorType generated_vector = make_with_random_values<VectorType>(
      make_not_null(&gen), make_not_null(&dist), VectorType{size});
  VectorType ref_generated_vector;
  ref_generated_vector.set_data_ref(&generated_vector);
  const VectorType larger_generated_vector =
      make_with_random_values<VectorType>(
          make_not_null(&gen), make_not_null(&dist), VectorType{size + 1});
  // each of the following options should error, the reference should have
  // received the wrong size
  if (test_kind == RefSizeErrorTestKind::Copy) {
    ref_generated_vector = larger_generated_vector;
  }
  if (test_kind == RefSizeErrorTestKind::ExpressionAssign) {
    ref_generated_vector = (larger_generated_vector + larger_generated_vector);
  }
  if (test_kind == RefSizeErrorTestKind::Move) {
    ref_generated_vector = std::move(larger_generated_vector);
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief tests a small sample of math functions after a move of a
/// `VectorType` initialized with `ValueType`
template <typename VectorType, typename ValueType>
void vector_test_math_after_move(
    tt::get_fundamental_type_t<ValueType> low =
        tt::get_fundamental_type_t<ValueType>{-100.0},
    tt::get_fundamental_type_t<ValueType> high =
        tt::get_fundamental_type_t<ValueType>{100.0}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<ValueType>> dist{low,
                                                                        high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  const size_t size = sdist(gen);
  const auto generated_value1 = make_with_random_values<ValueType>(
      make_not_null(&gen), make_not_null(&dist));
  const auto generated_value2 = make_with_random_values<ValueType>(
      make_not_null(&gen), make_not_null(&dist));
  const auto sum_generated_values = generated_value1 + generated_value2;
  const auto difference_generated_values = generated_value1 - generated_value2;

  const VectorType vector_math_lhs{size, generated_value1},
      vector_math_rhs{size, generated_value2};
  {
    INFO("Check move assignment and use after move");
    VectorType from_vector = make_with_random_values<VectorType>(
        make_not_null(&gen), make_not_null(&dist), VectorType{size});
    VectorType to_vector{};
    to_vector = std::move(from_vector);
    to_vector = vector_math_lhs + vector_math_rhs;
    CHECK_ITERABLE_APPROX(to_vector, (VectorType{size, sum_generated_values}));
    // clang-tidy: use after move (intentional here)
    CHECK(from_vector.size() == 0);  // NOLINT
    CHECK(from_vector.is_owning());
    from_vector = vector_math_lhs - vector_math_rhs;
    CHECK_ITERABLE_APPROX(from_vector,
                          (VectorType{size, difference_generated_values}));
    CHECK_ITERABLE_APPROX(to_vector, (VectorType{size, sum_generated_values}));
  }
  {
    INFO("Check move assignment and value of target")
    auto from_value = make_with_random_values<ValueType>(make_not_null(&gen),
                                                         make_not_null(&dist));
    VectorType from_vector{size, from_value};
    VectorType to_vector{};
    to_vector = std::move(from_vector);
    from_vector = vector_math_lhs + vector_math_rhs;
    CHECK_ITERABLE_APPROX(to_vector, (VectorType{size, from_value}));
    CHECK_ITERABLE_APPROX(from_vector,
                          (VectorType{size, sum_generated_values}));
  }
  {
    INFO("Check move constructor and use after move")
    VectorType from_vector = make_with_random_values<VectorType>(
        make_not_null(&gen), make_not_null(&dist), VectorType{size});
    VectorType to_vector{std::move(from_vector)};
    to_vector = vector_math_lhs + vector_math_rhs;
    CHECK(to_vector.size() == size);
    CHECK_ITERABLE_APPROX(to_vector, (VectorType{size, sum_generated_values}));
    // clang-tidy: use after move (intentional here)
    CHECK(from_vector.size() == 0);  // NOLINT
    CHECK(from_vector.is_owning());
    from_vector = vector_math_lhs - vector_math_rhs;
    CHECK_ITERABLE_APPROX(from_vector,
                          (VectorType{size, difference_generated_values}));
    CHECK_ITERABLE_APPROX(to_vector, (VectorType{size, sum_generated_values}));
  }

  {
    INFO("Check move constructor and value of target")
    auto from_value = make_with_random_values<ValueType>(make_not_null(&gen),
                                                         make_not_null(&dist));
    VectorType from_vector{size, from_value};
    const VectorType to_vector{std::move(from_vector)};
    from_vector = vector_math_lhs + vector_math_rhs;
    CHECK_ITERABLE_APPROX(to_vector, (VectorType{size, from_value}));
    CHECK_ITERABLE_APPROX(from_vector,
                          (VectorType{size, sum_generated_values}));
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Type alias to be more expressive with distribution bounds in vector
/// tests which call the generic math test below
using Bound = std::array<double, 2>;

/// \ingroup TestingFrameworkGroup
/// \brief the set of test types that may be used for the math operations
///
/// \details Three types of test are provided:
/// - `Normal` is used to indicate those tests which should be performed over
///   all combinations of the supplied vector type(s) and their value
///   types. This is useful for e.g. `+`.
///
/// - `Strict` is used to indicate those tests which should be performed over
///   only sets of the vector type and compared to the same operation of the set
///   of its value type. This is useful for e.g. `atan2`, which cannot take a
///   `DataVector` and a double as arguments.
///
/// - `Inplace` is used to indicate those tests which should be performed
///   maintaining the type of the left-hand side of the operator and not
///   including it in the combinations. Inplace operators such as `+=` have a
///   more restrictive condition on the type of the left hand side than do
///   simply `+`. (e.g. `double + complex<double>` compiles, but
///   `double += complex<double>` does not)
///
/// - `GivenOrderOfArgumentsOnly` is used to indicate that the arguments given
///   should not be taken in any combination apart from the given combination.
///   This should be used for highly restrictive operations which are only
///   supported for certain type combinations.
enum TestKind { Normal, Strict, Inplace, GivenOrderOfArgumentsOnly };

namespace detail {

// to choose between possible reference wrappers for the math tests.
namespace UseRefWrap {
struct Cref {};
struct None {};
struct Ref {};
}  // namespace UseRefWrap

// Wrap is used to wrap values in a std::reference_wrapper using std::cref and
// std::ref, or to not wrap at all. This is done to verify that all math
// operations work transparently with a `std::reference_wrapper` too.
template <class T>
decltype(auto) wrap(UseRefWrap::Cref /*meta*/, T& t) noexcept {
  return std::cref(t);
}

template <class T>
decltype(auto) wrap(UseRefWrap::None /*meta*/, T& t) noexcept {
  return t;
}

template <class T>
decltype(auto) wrap(UseRefWrap::Ref /*meta*/, T& t) noexcept {
  return std::ref(t);
}

using NonConstWrapperList = tmpl::list<UseRefWrap::None, UseRefWrap::Ref>;

using WrapperList = tmpl::push_front<NonConstWrapperList, UseRefWrap::Cref>;

// struct used for determining the full number of elements in a vector, array of
// vectors. Designed for use with `test_element_wise_function`
struct VectorOrArraySize {
  // array of vectors version, assumes all vectors in the array are the same
  // size, as will be the case for the relevant test helpers in this detail
  // namespace
  template <typename T, size_t S>
  size_t operator()(const std::array<T, S>& container) const noexcept {
    return S * VectorOrArraySize{}(container[0]);
  }
  // vector version
  template <typename T,
            Requires<cpp17::is_arithmetic_v<typename T::ElementType> or
                     tt::is_complex_of_fundamental_v<typename T::ElementType>> =
                nullptr>
  size_t operator()(const T& container) const noexcept {
    return container.size();
  }
};

// struct used for obtaining an indexed value in a vector, array of vectors.
// Designed for use with `test_element_wise_function`
struct VectorOrArrayAt {
  // array of vectors version
  template <typename T, size_t S>
  decltype(auto) operator()(std::array<T, S>& container,
                            size_t index) noexcept {
    return VectorOrArrayAt{}(gsl::at(container, index % S), index / S);
  }
  template <typename T, size_t S>
  decltype(auto) operator()(const std::array<T, S>& container,
                            size_t index) noexcept {
    return VectorOrArrayAt{}(gsl::at(container, index % S), index / S);
  }
  // vector version
  template <typename T,
            Requires<cpp17::is_arithmetic_v<typename T::ElementType> or
                     tt::is_complex_of_fundamental_v<typename T::ElementType>> =
                nullptr>
  decltype(auto) operator()(T& container, size_t index) noexcept {
    return container.at(index);
  }
};

// given an explicit template parameter pack `Wraps`, wrap the elements of
// `operand`, passed by pointer, element-by-element, and return the
// resulting tuple of wrapped elements.
template <typename... Wraps, typename... Operands, size_t... Is>
auto wrap_tuple(std::tuple<Operands...>& operand_values,
                std::index_sequence<Is...> /*meta*/) noexcept {
  return std::make_tuple(wrap(Wraps{}, get<Is>(operand_values))...);
}

// given the set of types of operands to test (`Operands`), and a set of
// reference wrappers (`Wraps`), make each operand with random values according
// to the bound from `Bounds`. Then, call
// `CHECK_CUSTOM_ELEMENT_WISE_FUNCTION_APPROX` to test the element wise `func`.
template <typename Function, typename... Bounds, typename... Wraps,
          typename... Operands, size_t... Is>
void test_function_on_vector_operands(
    const Function& function, const std::tuple<Bounds...>& bounds,
    const std::tuple<Wraps...>& /*wraps*/,
    const std::tuple<Operands...>& /*operands*/,
    std::index_sequence<Is...> /*meta*/) noexcept {
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<size_t> size_distribution{2, 5};
  const size_t size = size_distribution(generator);
  // using each distribution, generate a value for the appropriate operand type
  // and put it in a tuple.
  auto operand_values = std::make_tuple(make_with_random_values<Operands>(
      make_not_null(&generator),
      UniformCustomDistribution<
          tt::get_fundamental_type_t<get_vector_element_type_t<Operands>>>{
          std::get<Is>(bounds)},
      size)...);
  // wrap the tuple of random values according to the passed in `Wraps`
  auto wrapped_operands = wrap_tuple<Wraps...>(
      operand_values, std::make_index_sequence<sizeof...(Bounds)>{});
  CHECK_CUSTOM_ELEMENT_WISE_FUNCTION_APPROX(
      function, make_not_null(&wrapped_operands), VectorOrArrayAt{},
      VectorOrArraySize{});
}

// Set of structs for choosing between execution paths when assembling the
// argument list for the math testing utility
namespace TestFunctionsWithVectorArgumentsChoices {
struct FirstInplace {};
struct Continuing {};
struct Done {};
struct GivenOrderOfArgumentsOnly {};
}  // namespace TestFunctionsWithVectorArgumentsChoices

// helper struct implementation for choosing the next branch for assembling the
// argument list for the math testing utility. This is done to use pattern
// matching instead of SFINAE.
template <bool IsDone, bool IsFirstAssignment, bool GivenOrderOfArgumentsOnly>
struct next_choice_for_test_functions_recursion_impl;

// if the size of the vector list is the same as the size of the distribution
// bound list, then assembly is done
template <bool IsFirstAssignment, bool GivenOrderOfArgumentsOnly>
struct next_choice_for_test_functions_recursion_impl<
    true, IsFirstAssignment, GivenOrderOfArgumentsOnly> {
  using type = TestFunctionsWithVectorArgumentsChoices::Done;
};

// if we are assembling the first argument and the test type is Inplace, then
// treat it separately
template <>
struct next_choice_for_test_functions_recursion_impl<false, true, false> {
  using type = TestFunctionsWithVectorArgumentsChoices::FirstInplace;
};

// otherwise, continue assembling as ordinary
template <>
struct next_choice_for_test_functions_recursion_impl<false, false, false> {
  using type = TestFunctionsWithVectorArgumentsChoices::Continuing;
};

// if using mode GivenOrderOfArgumentsOnly, follow a distinct exection path with
// only the wraps assembled in combinations
template <bool IsFirstAssignment>
struct next_choice_for_test_functions_recursion_impl<false, IsFirstAssignment,
                                                     true> {
  using type =
      TestFunctionsWithVectorArgumentsChoices::GivenOrderOfArgumentsOnly;
};

// helper function for easily determining the next branch of the call operator
// for TestFunctionsWithVectorArgumentsImpl
template <typename VectorList, typename BoundList, TestKind Test>
using next_choice_for_test_functions_recursion =
    typename next_choice_for_test_functions_recursion_impl<
        tmpl::size<VectorList>::value == tmpl::size<BoundList>::value,
        tmpl::size<VectorList>::value == 0 and Test == TestKind::Inplace,
        Test == TestKind::GivenOrderOfArgumentsOnly>::type;

// functions to recursively assemble the arguments and wrappers for
// calling the operator tester `test_function_on_vector_operands`.
//
// `UniqueTypeList` is a tmpl::list which stores the set of unique types to
// test the functions with.
//
// `FirstOperand` is the first operand type, used when the Test is
// `TestKind::Inplace`, as inplace operators have specific demands on the
// first operand.

// base case: the correct number of operand types has been obtained in
// `Operands`, so this calls the test function with the function, bounds,
// reference wrap, and operand type information.
template <TestKind Test, typename UniqueTypeList, typename FirstOperand,
          typename... Operands, typename Function, typename... DistBounds,
          typename... Wraps>
void assemble_test_function_arguments_and_execute_tests(
    const Function& function, const std::tuple<DistBounds...>& bounds,
    const std::tuple<Wraps...>& wraps, const std::tuple<Operands...> operands,
    TestFunctionsWithVectorArgumentsChoices::Done /*meta*/) noexcept {
  test_function_on_vector_operands(
      function, bounds, wraps, operands,
      std::make_index_sequence<sizeof...(DistBounds)>{});
}

// general case: add an additional reference wrapper identification type from
// `WrapperList` to `wraps`, and an additional type from `UniqueTypeList`
// to `operands`, and recurse on each option.
template <TestKind Test, typename UniqueTypeList, typename FirstOperand,
          typename... Operands, typename Function, typename... DistBounds,
          typename... Wraps>
void assemble_test_function_arguments_and_execute_tests(
    const Function& function, const std::tuple<DistBounds...>& bounds,
    const std::tuple<Wraps...>& wraps, const std::tuple<Operands...>&& operands,
    TestFunctionsWithVectorArgumentsChoices::Continuing
    /*meta*/) noexcept {
  tmpl::for_each<WrapperList>([&function, &bounds, &wraps,
                               &operands ](const auto x) noexcept {
    tmpl::for_each<UniqueTypeList>([&function, &bounds, &wraps,
                                    &operands ](const auto y) noexcept {
      using next_vector = typename decltype(y)::type;
      using next_wrap = typename decltype(x)::type;
      assemble_test_function_arguments_and_execute_tests<Test, UniqueTypeList,
                                                         FirstOperand>(
          function, bounds, std::tuple_cat(wraps, std::tuple<next_wrap>{}),
          std::tuple_cat(operands, std::tuple<next_vector>{}),
          detail::next_choice_for_test_functions_recursion<
              tmpl::list<Wraps..., next_wrap>, tmpl::list<DistBounds...>,
              Test>{});
    });
  });
}

// case of first operand and inplace test: the left hand operand for inplace
// tests cannot be const, so the reference wrapper must be chosen
// accordingly. Also, the left hand size type is fixed to be FirstOperand.
template <TestKind Test, typename UniqueTypeList, typename FirstOperand,
          typename... Operands, typename Function, typename... DistBounds,
          typename... Wraps>
void assemble_test_function_arguments_and_execute_tests(
    const Function& function, const std::tuple<DistBounds...>& bounds,
    const std::tuple<Wraps...>& /*wraps*/,
    const std::tuple<Operands...>&& /*operands*/,
    TestFunctionsWithVectorArgumentsChoices::FirstInplace /*meta*/) noexcept {
  tmpl::for_each<NonConstWrapperList>([&function,
                                       &bounds ](const auto x) noexcept {
    using next_wrap = typename decltype(x)::type;
    assemble_test_function_arguments_and_execute_tests<Test, UniqueTypeList,
                                                       FirstOperand>(
        function, bounds, std::tuple<next_wrap>{}, std::tuple<FirstOperand>{},
        detail::next_choice_for_test_functions_recursion<
            tmpl::list<FirstOperand>, tmpl::list<DistBounds...>, Test>{});
  });
}

// case of TestKind of NoArgumentsCombinations: The Operands are already
// assembled, we just need to loop through the wrap possibilities.
template <TestKind Test, typename UniqueTypeList, typename FirstOperand,
          typename... Operands, typename Function, typename... DistBounds,
          typename... Wraps>
void assemble_test_function_arguments_and_execute_tests(
    const Function& function, const std::tuple<DistBounds...>& bounds,
    const std::tuple<Wraps...>& wraps, const std::tuple<Operands...>& operands,
    TestFunctionsWithVectorArgumentsChoices::
        GivenOrderOfArgumentsOnly /*meta*/) noexcept {
  tmpl::for_each<NonConstWrapperList>(
      [&function, &bounds, &wraps, &operands ](const auto x) noexcept {
        using next_wrap = typename decltype(x)::type;
        assemble_test_function_arguments_and_execute_tests<Test, UniqueTypeList,
                                                           FirstOperand>(
            function, bounds, std::tuple_cat(wraps, std::tuple<next_wrap>{}),
            operands,
            detail::next_choice_for_test_functions_recursion<
                tmpl::list<Wraps..., next_wrap>, tmpl::list<DistBounds...>,
                Test>{});
      });
}

// dispatch function for the individual tuples of functions and bounds for their
// respective distributions. This processes the requested set of vector types to
// test and passes the resulting request to the recursive argument assembly
// function assemble_test_function_arguments_and_execute_tests above
template <TestKind Test, typename VectorType0, typename... VectorTypes,
          typename Function, typename... DistBounds>
void test_function_with_vector_arguments_impl(
    const std::tuple<Function, std::tuple<DistBounds...>>&
        function_and_argument_bounds) noexcept {
  // The unique set of possible operands
  using operand_type_list = tmpl::conditional_t<
      Test == TestKind::Strict,
      // if strict, the operand choices are just the vector types passed in
      tmpl::remove_duplicates<tmpl::list<VectorType0, VectorTypes...>>,
      // else, the operand choices include both the vectors and their element
      // types
      tmpl::remove_duplicates<tmpl::list<
          VectorType0, VectorTypes..., get_vector_element_type_t<VectorType0>,
          get_vector_element_type_t<VectorTypes>...>>>;

  using starting_operands =
      tmpl::conditional_t<Test == TestKind::GivenOrderOfArgumentsOnly,
                          std::tuple<VectorType0, VectorTypes...>,
                          std::tuple<>>;
  assemble_test_function_arguments_and_execute_tests<Test, operand_type_list,
                                                     VectorType0>(
      get<0>(function_and_argument_bounds) /*function*/,
      get<1>(function_and_argument_bounds) /*tuple of bounds*/, std::tuple<>{},
      starting_operands{},
      next_choice_for_test_functions_recursion<
          tmpl::list<>, tmpl::list<DistBounds...>, Test>{});
}
}  // namespace detail

/*!
 * \ingroup TestingFrameworkGroup
 * \brief General entry function for testing arbitrary math functions
 * on vector types
 *
 * \details This utility tests all combinations of the operator on the type
 * arguments, and all combinations of reference or constant reference wrappers
 * on all arguments. In certain test cases (see below), it also tests using the
 * vector type's `value_type`s in the operators as well (e.g. `DataVector +
 * double`). This is very useful for quickly generating a lot of tests, but the
 * number of tests scales exponentially in the number of arguments. Therefore,
 * functions with many arguments can be time-consuming to
 * run. 4-or-more-argument functions should be used only if completely necessary
 * and with caution. Any number of vector types may be specified, and tests are
 * run on all unique combinations of the provided. For instance, if only one
 * type is provided, the tests will be run only on combinations of that single
 * type and its `value_type`.
 *
 * \param tuple_of_functions_and_argument_bounds A tuple of tuples, in which
 *  the inner tuple contains first a function object followed by a tuple of
 *  2-element arrays equal to the number of arguments, which represent the
 *  bounds for the random generation of the respective arguments. This system
 *  is provided for robust testing of operators like `/`, where the left-hand
 *  side has a different valid set of values than the right-hand-side.
 *
 * \tparam Test from the `TestKind` enum, determines whether the tests will
 *  be:
 *  - `TestKind::Normal`: executed on all combinations of arguments and value
 *    types
 *  - `TestKind::Strict`: executed on all combinations of arguments, for only
 *    the vector types
 *  - `TestKind::Inplace`: executed on all combinations of arguments after the
 *    first, so first is always the 'left hand side' of the operator. In this
 *    case, at least two `VectorTypes` must be specified, where the first is
 *    used only for the left-hand side.
 *  - `TestKind::GivenOrderOfArgumentsOnly`: executed on only the combination of
 *    arguments provided, in the order provided. In this case, the number of
 *    provided types in `typename VectorType0, typename... VectorTypes` must
 *    precisely match the number of arguments taken by the function.
 *
 * \tparam VectorType0 The first vector type for which combinations are
 *  tested. The first is accepted as a separate template argument for
 *  appropriately handling `Inplace` tests.
 * \tparam VectorTypes The remaining types for which combinations are tested.
 *  Any number of types may be passed in, and the test will check the
 *  appropriate combinations of the vector types and (depending on the `Test`)
 *  the respective `value_type`s.
 */
template <TestKind Test, typename VectorType0, typename... VectorTypes,
          typename... FunctionsAndArgumentBounds>
void test_functions_with_vector_arguments(
    const std::tuple<FunctionsAndArgumentBounds...>&
        tuple_of_functions_and_argument_bounds) noexcept {
  tuple_fold(
      tuple_of_functions_and_argument_bounds,
      [](const auto& function_and_argument_bounds) noexcept {
        detail::test_function_with_vector_arguments_impl<Test, VectorType0,
                                                         VectorTypes...>(
            function_and_argument_bounds);
      });
}
}  // namespace VectorImpl
}  // namespace TestHelpers
