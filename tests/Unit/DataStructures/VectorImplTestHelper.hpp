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
#include "Utilities/DereferenceWrapper.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TmplDebugging.hpp"
#include "Utilities/Tuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/TestingFramework.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace TestHelpers {
namespace VectorImpl {

namespace detail {
// `check_vectors` checks equality of vectors and related data types for tests.
//
// Comparisons between any pairs of the following are supported:
// vector types
// arithmetic types
// arrays of vectors
// arrays of arithmetic types

// between two vectors
template <typename T1, typename T2, typename VT1, typename VT2>
inline void check_vectors(const ::VectorImpl<T1, VT1>& t1,
                          const ::VectorImpl<T2, VT2>& t2) noexcept {
  CHECK_ITERABLE_APPROX(VT1{t1}, VT2{t2});
}
// between two arithmetic types
template <typename T1, typename T2,
          Requires<cpp17::is_arithmetic_v<T1> and cpp17::is_arithmetic_v<T2>> =
              nullptr>
inline void check_vectors(const T1& t1, const T2& t2) noexcept {
  CHECK(approx(t1) == t2);
}
// between an arithmetic type and a vector
template <typename T1, typename T2, typename VT2,
          Requires<cpp17::is_arithmetic_v<T1>> = nullptr>
void check_vectors(const T1& t1, const ::VectorImpl<T2, VT2>& t2) noexcept {
  check_vectors(VT2{t2.size(), t1}, t2);
}
// between a vector and an arithmetic type
template <typename T1, typename VT1, typename T2,
          Requires<cpp17::is_arithmetic_v<T2>> = nullptr>
void check_vectors(const ::VectorImpl<T1, VT1>& t1, const T2& t2) noexcept {
  check_vectors(t2, t1);
}
// between two arrays
// between two arrays
template <typename T1, typename T2, size_t S>
void check_vectors(const std::array<T1, S>& t1,
                   const std::array<T2, S>& t2) noexcept {
  for (size_t i = 0; i < S; i++) {
    check_vectors(gsl::at(t1, i), gsl::at(t2, i));
  }
}
// between an array of vectors and an arithmetic type
template <typename T1, typename T2, size_t S>
void check_vectors(const std::array<T1, S>& t1, const T2& t2) noexcept {
  for (const auto& array_element : t1) {
    check_vectors(array_element, t2);
  }
}
// between an arithmetic type and an array of vectors
template <typename T1, typename T2, size_t S>
void check_vectors(const T1& t1, const std::array<T2, S>& t2) noexcept {
  check_vectors(t2, t1);
}
} // namespace detail

/// \ingroup TestingFrameworkGroup
/// \brief test construction and assignment of a `VectorType` with a `ValType`
template <typename VectorType, typename ValType>
void vector_test_construct_and_assign(
    typename tt::get_fundamental_type_t<ValType> low =
        typename tt::get_fundamental_type_t<ValType>{-100},
    typename tt::get_fundamental_type_t<ValType> high =
        typename tt::get_fundamental_type_t<ValType>{100}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<typename tt::get_fundamental_type_t<ValType>> dist{
      low, high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  size_t size = sdist(gen);

  VectorType size_constructed{size};
  CHECK(size_constructed.size() == size);
  auto generated_val1 = make_with_random_values<ValType>(make_not_null(&gen),
                                                         make_not_null(&dist));

  VectorType value_size_constructed{size, generated_val1};
  CHECK(value_size_constructed.size() == size);
  std::for_each(
      value_size_constructed.begin(), value_size_constructed.end(),
      [generated_val1](typename VectorType::value_type element) noexcept {
        CHECK(element == generated_val1);
      });

  // random generation must use `make_with_random_values`, because stored value
  // in vector type might be a non-fundamental type.
  auto generated_val2 = make_with_random_values<ValType>(make_not_null(&gen),
                                                         make_not_null(&dist)),
       generated_val3 = make_with_random_values<ValType>(make_not_null(&gen),
                                                         make_not_null(&dist));

  VectorType initializer_list_constructed{
      {static_cast<typename VectorType::value_type>(generated_val2),
       static_cast<typename VectorType::value_type>(generated_val3)}};
  CHECK(initializer_list_constructed.size() == 2);
  CHECK(initializer_list_constructed.is_owning());
  CHECK(gsl::at(initializer_list_constructed, 0) == generated_val2);
  CHECK(gsl::at(initializer_list_constructed, 1) == generated_val3);

  typename VectorType::value_type raw_ptr[2] = {generated_val2, generated_val3};

  VectorType pointer_size_constructed{
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

  VectorType move_constructed{std::move(move_assignment_initialized)};
  CHECK(move_constructed.is_owning());
  CHECK(move_constructed == pointer_size_constructed);

  // clang-tidy has performance complaints, and we're checking functionality
  VectorType copy_constructed{move_constructed};  // NOLINT
  CHECK(copy_constructed.is_owning());
  CHECK(copy_constructed == pointer_size_constructed);
}

/// \ingroup TestingFrameworkGroup
/// \brief test the serialization of a `VectorType` constructed with a `ValType`
template <typename VectorType, typename ValType>
void vector_test_serialize(
    typename tt::get_fundamental_type_t<ValType> low =
        typename tt::get_fundamental_type_t<ValType>{-100},
    typename tt::get_fundamental_type_t<ValType> high =
        typename tt::get_fundamental_type_t<ValType>{100}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<typename tt::get_fundamental_type_t<ValType>> dist{
      low, high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  const size_t size = sdist(gen);
  VectorType vector_test{size}, vector_control{size};
  VectorType vector_ref;
  auto start_val = make_with_random_values<ValType>(make_not_null(&gen),
                                                    make_not_null(&dist));
  auto val_diff = make_with_random_values<ValType>(make_not_null(&gen),
                                                   make_not_null(&dist));
  // generate_series is used to generate a pair of equivalent, but independently
  // constructed, data sets to fill the vectors with.
  ValType current_val = start_val;
  auto generate_series = [&current_val, val_diff ]() noexcept {
    return current_val += val_diff;
  };
  std::generate(vector_test.begin(), vector_test.end(), generate_series);
  current_val = start_val;
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
  vector_ref.set_data_ref(make_not_null(&vector_test));
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
/// \brief test the construction and movement of a reference `VectorType`
/// constructed with a `ValType`
template <typename VectorType, typename ValType>
void vector_test_ref(typename tt::get_fundamental_type_t<ValType> low =
                         typename tt::get_fundamental_type_t<ValType>{-100},
                     typename tt::get_fundamental_type_t<ValType> high =
                         typename tt::get_fundamental_type_t<ValType>{
                             100}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<typename tt::get_fundamental_type_t<ValType>> dist{
      low, high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  size_t size = sdist(gen);

  VectorType original_vector(size);
  fill_with_random_values(make_not_null(&original_vector), make_not_null(&gen),
                          make_not_null(&dist));

  SECTION(
      "Check construction, copy, move, and ownership of reference vectors") {
    VectorType ref_vector;
    ref_vector.set_data_ref(make_not_null(&original_vector));
    CHECK_FALSE(ref_vector.is_owning());
    CHECK(original_vector.is_owning());
    CHECK(ref_vector.data() == original_vector.data());

    VectorType data_check{original_vector};
    CHECK(ref_vector.size() == size);
    CHECK(ref_vector == data_check);
    test_copy_semantics(ref_vector);

    VectorType ref_vector_copy;
    ref_vector_copy.set_data_ref(make_not_null(&ref_vector));
    test_move_semantics(std::move(ref_vector), ref_vector_copy);
    VectorType move_assignment_initialized;
    move_assignment_initialized = std::move(ref_vector_copy);
    CHECK(not move_assignment_initialized.is_owning());
    VectorType move_constructed{std::move(move_assignment_initialized)};
    CHECK(not move_constructed.is_owning());
  }
  SECTION("Check movement acts appropriately on both source and target refs") {
    VectorType ref_original_vector;
    ref_original_vector.set_data_ref(make_not_null(&original_vector));
    VectorType generated_vector(size);
    fill_with_random_values(make_not_null(&generated_vector),
                            make_not_null(&gen), make_not_null(&dist));
    VectorType generated_vector_copy = generated_vector;
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
    VectorType data_check_vector;
    // clang-tidy: false positive, used after it was moved
    data_check_vector = ref_original_vector;  // NOLINT
    CHECK(data_check_vector == generated_vector_copy);
  }
  SECTION("Check math affects both data vectors which share a ref") {
    auto generated_val1 = make_with_random_values<ValType>(
             make_not_null(&gen), make_not_null(&dist)),
         generated_val2 = make_with_random_values<ValType>(
             make_not_null(&gen), make_not_null(&dist));
    auto sum_generated_vals = generated_val1 + generated_val2;
    VectorType sharing_vector{size, generated_val1};
    VectorType owning_vector{size, generated_val2};
    sharing_vector.set_data_ref(make_not_null(&owning_vector));
    sharing_vector = sharing_vector + generated_val1;
    detail::check_vectors(owning_vector, sum_generated_vals);
    detail::check_vectors(sharing_vector, sum_generated_vals);
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief test that `VectorType` appropriately errors out when assigned to
/// another `VectorType` with the wrong size that is non-owning.
///
/// \details a calling function should be an `ASSERTION_TEST()` and check for
/// the string "Must copy into same size".
template <typename VectorType,
          typename ValType = typename VectorType::ElementType>
void vector_ref_test_size_error(
    typename tt::get_fundamental_type_t<ValType> low =
        typename tt::get_fundamental_type_t<ValType>{-100},
    typename tt::get_fundamental_type_t<ValType> high =
        typename tt::get_fundamental_type_t<ValType>{100}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<typename tt::get_fundamental_type_t<ValType>> dist{
      low, high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  size_t size = sdist(gen);
  VectorType generated_vector{size, 0};
  fill_with_random_values(make_not_null(&generated_vector), make_not_null(&gen),
                          make_not_null(&dist));
  VectorType ref_generated_vector;
  ref_generated_vector.set_data_ref(make_not_null(&generated_vector));
  VectorType larger_generated_vector{size + 1, 0};
  fill_with_random_values(make_not_null(&larger_generated_vector),
                          make_not_null(&gen), make_not_null(&dist));
  // this line should error, the reference should have received the smaller size
  ref_generated_vector = larger_generated_vector;
}

/// \ingroup TestingFrameworkGroup
/// \brief test that reference to a `VectorType` appropriately errors out when
/// moved to another `VectorType` with the wrong size that is non-owning.
///
/// \details a calling function should be an `ASSERTION_TEST()` and check for
/// the string "Must copy into same size".
template <typename VectorType,
          typename ValType = typename VectorType::ElementType>
void vector_ref_test_move_size_error(
    typename tt::get_fundamental_type_t<ValType> low =
        typename tt::get_fundamental_type_t<ValType>{-100},
    typename tt::get_fundamental_type_t<ValType> high =
        typename tt::get_fundamental_type_t<ValType>{100}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<typename tt::get_fundamental_type_t<ValType>> dist{
      low, high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  size_t size = sdist(gen);
  VectorType generated_vector{size, 0};
  fill_with_random_values(make_not_null(&generated_vector), make_not_null(&gen),
                          make_not_null(&dist));
  VectorType ref_generated_vector;
  ref_generated_vector.set_data_ref(make_not_null(&generated_vector));
  VectorType larger_generated_vector{size + 1, 0};
  fill_with_random_values(make_not_null(&larger_generated_vector),
                          make_not_null(&gen), make_not_null(&dist));
  // this line should error, the reference should have received the smaller size
  ref_generated_vector = std::move(larger_generated_vector);
}

/// \ingroup TestingFrameworkGroup
/// \brief tests a small sample of math functions after movement of a
/// `VectorType` initialized with `ValType`
template <typename VectorType, typename ValType>
void vector_test_math_after_move(
    typename tt::get_fundamental_type_t<ValType> low =
        typename tt::get_fundamental_type_t<ValType>{-100},
    typename tt::get_fundamental_type_t<ValType> high =
        typename tt::get_fundamental_type_t<ValType>{100}) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<typename tt::get_fundamental_type_t<ValType>> dist{
      low, high};
  UniformCustomDistribution<size_t> sdist{2, 20};

  size_t size = sdist(gen);
  auto generated_val1 = make_with_random_values<ValType>(make_not_null(&gen),
                                                         make_not_null(&dist)),
       generated_val2 = make_with_random_values<ValType>(make_not_null(&gen),
                                                         make_not_null(&dist));
  ValType sum_generated_vals = generated_val1 + generated_val2,
          difference_generated_vals = generated_val1 - generated_val2;

  VectorType vector_math_lhs{size, generated_val1},
      vector_math_rhs{size, generated_val2};
  SECTION("Check move assignment and use after move") {
    VectorType source_vector{size, 0};
    fill_with_random_values(make_not_null(&source_vector), make_not_null(&gen),
                            make_not_null(&dist));
    VectorType target_vector{};
    target_vector = std::move(source_vector);
    target_vector = vector_math_lhs + vector_math_rhs;
    detail::check_vectors(target_vector, VectorType{size, sum_generated_vals});
    // clang-tidy: use after move (intentional here)
    CHECK(source_vector.size() == 0);  // NOLINT
    CHECK(source_vector.is_owning());
    source_vector = vector_math_lhs - vector_math_rhs;
    detail::check_vectors(source_vector,
                          VectorType{size, difference_generated_vals});
    detail::check_vectors(target_vector, sum_generated_vals);
  }

  SECTION("Check move assignment and value of target") {
    auto source_val = make_with_random_values<ValType>(make_not_null(&gen),
                                                       make_not_null(&dist));
    VectorType source_vector{size, source_val};
    VectorType target_vector{};
    target_vector = std::move(source_vector);
    source_vector = vector_math_lhs + vector_math_rhs;
    detail::check_vectors(target_vector, source_val);
    detail::check_vectors(source_vector, sum_generated_vals);
  }

  SECTION("Check move constructor and use after move") {
    VectorType source_vector{size, 0};
    fill_with_random_values(make_not_null(&source_vector), make_not_null(&gen),
                            make_not_null(&dist));
    VectorType target_vector{std::move(source_vector)};
    target_vector = vector_math_lhs + vector_math_rhs;
    CHECK(target_vector.size() == size);
    detail::check_vectors(target_vector, sum_generated_vals);
    // clang-tidy: use after move (intentional here)
    CHECK(source_vector.size() == 0);  // NOLINT
    CHECK(source_vector.is_owning());
    source_vector = vector_math_lhs - vector_math_rhs;
    detail::check_vectors(source_vector,
                          VectorType{size, difference_generated_vals});
    detail::check_vectors(target_vector, VectorType{size, sum_generated_vals});
  }

  SECTION("Check move constructor and value of target") {
    auto source_val = make_with_random_values<ValType>(make_not_null(&gen),
                                                       make_not_null(&dist));
    VectorType source_vector{size, source_val};
    VectorType target_vector{std::move(source_vector)};
    source_vector = vector_math_lhs + vector_math_rhs;
    detail::check_vectors(target_vector, VectorType{size, source_val});
    detail::check_vectors(source_vector, VectorType{size, sum_generated_vals});
  }
}
struct UseRefWrapNone {};
struct UseRefWrapCref {};
struct UseRefWrapRef {};

// Wrap is used to wrap values in a std::reference_wrapper using std::cref and
// std::ref, or to not wrap at all. This is done to verify that all math
// operations work transparently with a `std::reference_wrapper` too.
/// \cond HIDDEN_SYMBOLS
template <typename Wrap, class T,
          Requires<cpp17::is_same_v<Wrap, UseRefWrapCref>> = nullptr>
decltype(auto) wrap(T& t) noexcept {
  return std::cref(t);
}

template <typename Wrap, class T,
          Requires<cpp17::is_same_v<Wrap, UseRefWrapRef>> = nullptr>
decltype(auto) wrap(T& t) noexcept {
  return std::ref(t);
}

template <typename Wrap, class T,
          Requires<cpp17::is_same_v<Wrap, UseRefWrapNone>> = nullptr>
decltype(auto) wrap(T& t) noexcept {
  return t;
}
/// \endcond

using Bound = std::array<double, 2>;

using WrapperList = tmpl::list<UseRefWrapCref, UseRefWrapRef, UseRefWrapNone>;
using NonConstWrapperList = tmpl::list<UseRefWrapRef, UseRefWrapNone>;

/// \ingroup TestingFrameworkGroup
/// \brief the set of test types that may be used for the math operations
///
/// \details Three types of test are provided:
/// `Normal` is used to indicate those tests which should be performed over all
/// combinations of the supplied vector type(s) and their value types. This is
/// useful for e.g. `+`.
///
/// `Strict` is used to indicate those tests which should be performed over only
/// sets of the vector type and compared to the same operation of the set of its
/// value type. This is useful for e.g. `atan2`, which cannot take a
/// `DataVector` and a double as arguments.
///
/// `Inplace` is used to indicate those tests which should be performed
/// maintaining the lhs type and not including it in the combinations. Inplace
/// operators such as `+=` have a more restrictive condition on the type of the
/// left hand side than do simply `+`. (e.g. `double + complex<double>`
/// compiles, but `double+=complex<double>` does not)
enum TestType { Normal, Strict, Inplace };

namespace detail {

// struct used for determining the full number of elements in a vector, array of
// vectors, or individual element (which is always 1). Created for use with
// `test_element_wise_function`
struct VectorOrArraySize {
  template <typename T, size_t S>
  size_t operator()(std::array<T, S> container) noexcept {
    return S * VectorOrArraySize{}(container[0]);
  }

  template <typename T,
            Requires<cpp17::is_arithmetic_v<typename T::ElementType>> = nullptr>
  size_t operator()(T container) noexcept {
    return container.size();
  }

  template <typename T, Requires<cpp17::is_arithmetic_v<T>> = nullptr>
  size_t operator()(T /*value*/) noexcept {
    return 1;
  }
};

// struct used for obtaining an indexed value in a vector, array of vectors, or
// individual element (which just returns the element regardless of
// index). Created for use with `test_element_wise_function`
struct VectorOrArrayAt {
  template <typename T, size_t S>
  auto operator()(std::array<T, S> container, size_t index) noexcept {
    return VectorOrArrayAt{}(gsl::at(container, index % S), index / S);
  }

  template <typename T,
            Requires<cpp17::is_arithmetic_v<typename T::ElementType>> = nullptr>
  auto operator()(T container, size_t index) noexcept {
    return container.at(index);
  }

  template <typename T, Requires<cpp17::is_arithmetic_v<T>> = nullptr>
  auto operator()(T value, size_t /*index*/) noexcept {
    return value;
  }
};

// given an explicit template parameter pack `Wraps`, wrap the elements of
// `operand_tup`, passed by pointer, element-by-element, and return the
// resulting tuple of wrapped elements.
template <typename... Wraps, typename... Operands, size_t... Is>
auto wrap_tuple(gsl::not_null<std::tuple<Operands...>*> operand_tup,
                std::index_sequence<Is...> /*meta*/) noexcept {
  return std::make_tuple(wrap<Wraps>(get<Is>(*operand_tup))...);
}

// given the set of types of operands to test (`OperandTypes`), and a set of
// reference wrappers (`Wraps`), make each operand with random values according
// to the bound from `Bounds`. Then, call
// `CHECK_CUSTOM_ELEMENT_WISE_FUNCTION_APPROX` to test the element wise `func`.
template <typename Functor, typename... Bounds, typename... Wraps,
          typename... OperandTypes, size_t... Is>
void vector_test_operator(Functor func, std::tuple<Bounds...> bound_tup,
                          std::tuple<Wraps...> /*wraps*/,
                          std::tuple<OperandTypes...> /*operands*/,
                          std::index_sequence<Is...> /*meta*/) noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{2, 5};
  DataVector used_for_size{sdist(gen)};
  auto val_tup = std::make_tuple(make_with_random_values<OperandTypes>(
      make_not_null(&gen),
      UniformCustomDistribution<typename tt::get_fundamental_type_t<
          vector_base_type_t<OperandTypes>>>{std::get<Is>(bound_tup)},
      used_for_size)...);
  auto wrapped_tup = wrap_tuple<Wraps...>(
      make_not_null(&val_tup), std::make_index_sequence<sizeof...(Bounds)>{});
  CHECK_CUSTOM_ELEMENT_WISE_FUNCTION_APPROX(
      func, wrapped_tup, VectorOrArrayAt{}, VectorOrArraySize{});
}

// dispatch function to recursively assemble the arguments and wrappers for
// calling the operator tester `vector_test_operator`.
//
// `UniqueTypeList` is a tmpl::list which stores the set of unique types to test
// the functions with.
//
// `VectorType0` is the first operand type, used when the Test is
// `TestType::Inplace`, as inplace operators have specific demands on the first
// operand.
template <TestType Test, typename UniqueTypeList, typename VectorType0>
struct VectorTestFunctorsImpl {
  // base case: the correct number of operand types has been obtained in
  // `Vectors`, so this calls the test function with the function, bounds,
  // reference wrap, and operand type information.
  template <typename Functor, typename... DistBounds, typename... Wraps,
            typename... Vectors,
            Requires<sizeof...(Vectors) == sizeof...(DistBounds)> = nullptr>
  void operator()(Functor func, std::tuple<DistBounds...> bound_tup,
                  std::tuple<Wraps...> wrap_tup,
                  std::tuple<Vectors...> vec_tup) noexcept {
    vector_test_operator(func, bound_tup, wrap_tup, vec_tup,
                         std::make_index_sequence<sizeof...(DistBounds)>{});
  }
  // general case: add an additional reference wrapper identification type from
  // `WrapperList` to `wrap_tup`, and an additional type from `UniqueTypeList`
  // to `vec_tup`, and recurse on each option.
  template <
      typename Functor, typename... DistBounds, typename... Wraps,
      typename... Vectors,
      Requires<sizeof...(Vectors) != sizeof...(DistBounds)> = nullptr,
      Requires<sizeof...(Vectors) != 0 or Test != TestType::Inplace> = nullptr>
  void operator()(Functor func, std::tuple<DistBounds...> bound_tup,
                  std::tuple<Wraps...> wrap_tup,
                  std::tuple<Vectors...> vec_tup) noexcept {
    tmpl::for_each<WrapperList>([&func, &bound_tup, &wrap_tup,
                                 &vec_tup ](auto x) noexcept {
      tmpl::for_each<UniqueTypeList>([&func, &bound_tup, &wrap_tup,
                                      &vec_tup ](auto y) noexcept {
        VectorTestFunctorsImpl<Test, UniqueTypeList, VectorType0>{}(
            func, bound_tup,
            std::tuple_cat(wrap_tup, std::tuple<typename decltype(x)::type>{}),
            std::tuple_cat(vec_tup, std::tuple<typename decltype(y)::type>{}));
      });
    });
  }
  // case of first operand and inplace test: the left hand operand for inplace
  // tests cannot be const, so the reference wrapper must be chosen
  // accordingly. Also, the left hand size type is fixed to be VectorType0.
  template <
      typename Functor, typename... DistBounds, typename... Wraps,
      typename... Vectors,
      Requires<sizeof...(Vectors) != sizeof...(DistBounds)> = nullptr,
      Requires<sizeof...(Vectors) == 0 and Test == TestType::Inplace> = nullptr>
  void operator()(Functor func, std::tuple<DistBounds...> bound_tup,
                  std::tuple<Wraps...> /*wrap_tup*/,
                  std::tuple<Vectors...> /*vec_tup*/) noexcept {
    tmpl::for_each<NonConstWrapperList>([&func, &bound_tup ](auto x) noexcept {
      VectorTestFunctorsImpl<Test, UniqueTypeList, VectorType0>{}(
          func, bound_tup, std::tuple<typename decltype(x)::type>{},
          std::tuple<VectorType0>{});
    });
  }
};
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
 * the template parameters are:
 *
 * \tparam Test - from the `TestType` enum, determines whether the tests will
 * be:
 * - `TestType::Normal`: executed on all combinations of arguments and value
 *   types
 * - `TestType::Strict`: executed on all combinations of arguments, for only the
 *   vector types
 * - `TestType::Inplace`: executed on all combinations of arguments after the
 *   first, so first is always the 'left hand side' of the operator. In this
 *   case, at least two `VectorTypes` must be specified, where the first is used
 *   only for the left-hand side.
 *
 * \tparam VectorTypes - the types for which combinations are tested.  Any
 *  number of types may be passed in, and the test will check the appropriate
 *  combinations of the vector types and (depending on the `Test`) the
 *  respective `value_type`s.
 */
template <TestType Test, typename VectorType0, typename... VectorTypes>
struct VectorTestFunctors {
  /*!
   * \brief Entry function to the Functor-based Vector testing utility
   *
   * \param func_tup A tuple of tuples, in which the inner tuple contains first
   * a function object followed by a tuple of 2-element arrays equal to the
   * number of arguments, which represent the bounds for the random generation
   * of the respective arguments. This system is provided for robust testing of
   * operators like `/`, where the left-hand side has a different valid set of
   * values than the right-hand-side.
   */
  template <typename... FunctorsAndDistBounds>
  void operator()(std::tuple<FunctorsAndDistBounds...> func_tup) noexcept {
    tuple_fold(func_tup, [](auto fdata) noexcept {
      using operand_type_list = tmpl::conditional_t<
          Test == TestType::Strict,
          tmpl::remove_duplicates<tmpl::list<VectorType0, VectorTypes...>>,
          tmpl::remove_duplicates<tmpl::list<
              VectorType0, VectorTypes..., vector_base_type_t<VectorType0>,
              vector_base_type_t<VectorTypes>...>>>;
      detail::VectorTestFunctorsImpl<Test, operand_type_list, VectorType0>{}(
          get<0>(fdata), get<1>(fdata), std::tuple<>{}, std::tuple<>{});
    });
  }
};
}  // namespace VectorImpl
}  // namespace TestHelpers
