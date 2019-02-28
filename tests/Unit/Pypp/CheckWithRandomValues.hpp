// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

// IWYU pragma: begin_exports
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/comparison/equal.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/debug/assert.hpp>
#include <boost/preprocessor/list/adt.hpp>
#include <boost/preprocessor/list/fold_right.hpp>
#include <boost/preprocessor/list/for_each.hpp>
#include <boost/preprocessor/list/for_each_product.hpp>
#include <boost/preprocessor/list/to_tuple.hpp>
#include <boost/preprocessor/list/transform.hpp>
#include <boost/preprocessor/logical/compl.hpp>
#include <boost/preprocessor/logical/not.hpp>
#include <boost/preprocessor/punctuation/is_begin_parens.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/enum.hpp>
#include <boost/preprocessor/tuple/pop_front.hpp>
#include <boost/preprocessor/tuple/push_back.hpp>
#include <boost/preprocessor/tuple/push_front.hpp>
#include <boost/preprocessor/tuple/rem.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/tuple/to_array.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <boost/preprocessor/variadic/elem.hpp>
#include <boost/preprocessor/variadic/to_list.hpp>
#include <boost/preprocessor/variadic/to_tuple.hpp>
#include <boost/vmd/is_empty.hpp>
// IWYU pragma: end_exports

#include <initializer_list>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/Pypp/Pypp.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace pypp {
namespace TestHelpers_detail {
template <typename T>
using is_not_null = tt::is_a<gsl::not_null, T>;

template <typename T>
struct RemoveNotNull {
  using type = T;
};

template <typename T>
struct RemoveNotNull<gsl::not_null<T>> {
  using type = std::remove_pointer_t<T>;
};

template <typename MemberArg, typename UsedForSize, typename = std::nullptr_t>
struct ConvertToTensorImpl;

template <typename UsedForSize>
struct ConvertToTensorImpl<double, UsedForSize> {
  static auto apply(const double& value,
                    const UsedForSize& used_for_size) noexcept {
    return make_with_value<Scalar<DataVector>>(used_for_size, value);
  }
};

template <size_t Dim, typename UsedForSize>
struct ConvertToTensorImpl<std::array<double, Dim>, UsedForSize> {
  static auto apply(const std::array<double, Dim>& arr,
                    const UsedForSize& used_for_size) noexcept {
    auto array_as_tensor =
        make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.);
    for (size_t i = 0; i < Dim; ++i) {
      array_as_tensor.get(i) = gsl::at(arr, i);
    }
    return array_as_tensor;
  }
};

template <typename ReturnType, typename = std::nullptr_t>
struct ForwardToPyppImpl {
  template <typename MemberArg, typename UsedForSize>
  static decltype(auto) apply(const MemberArg& member_arg,
                              const UsedForSize& /*used_for_size*/) noexcept {
    return member_arg;
  }
};

template <typename ReturnType>
struct ForwardToPyppImpl<
    ReturnType,
    Requires<(tt::is_a_v<Tensor, ReturnType> or
              tt::is_std_array_v<ReturnType>)and cpp17::
                 is_same_v<typename ReturnType::value_type, DataVector>>> {
  template <typename MemberArg, typename UsedForSize>
  static decltype(auto) apply(const MemberArg& member_arg,
                              const UsedForSize& used_for_size) noexcept {
    return ConvertToTensorImpl<MemberArg, UsedForSize>::apply(member_arg,
                                                              used_for_size);
  }
};

// Given the member variable of type MemberArg (either a double or array of
// doubles), performs a conversion so that it can be correctly forwarded to
// Pypp. If ReturnType is not a Tensor or array of DataVectors,
// member_arg is simply forwarded, otherwise it is converted to a Tensor of
// DataVectors.
template <typename PyppReturn, typename MemberArg, typename UsedForSize>
decltype(auto) forward_to_pypp(const MemberArg& member_arg,
                               const UsedForSize& used_for_size) noexcept {
  return ForwardToPyppImpl<PyppReturn>::apply(member_arg, used_for_size);
}

template <class F, class T, class TagsList, class Klass, class... ReturnTypes,
          class... ArgumentTypes, class... MemberArgs, size_t... ResultIs,
          size_t... ArgumentIs, size_t... MemberArgsIs>
void check_with_random_values_impl(
    F&& f, const Klass& klass, const std::string& module_name,
    const std::vector<std::string>& function_names, std::mt19937 generator,
    std::array<std::uniform_real_distribution<>, sizeof...(ArgumentTypes)>
        distributions,
    const std::tuple<MemberArgs...>& member_args, const T& used_for_size,
    tmpl::list<ReturnTypes...> /*return_types*/,
    tmpl::list<ArgumentTypes...> /*argument_types*/,
    std::index_sequence<ResultIs...> /*index_return_types*/,
    std::index_sequence<ArgumentIs...> /*index_argument_types*/,
    std::index_sequence<MemberArgsIs...> /*index_member_args*/,
    // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
    TagsList /*meta*/, const double epsilon = 1.0e-12) {
  // Note: generator and distributions cannot be const.
  std::tuple<ArgumentTypes...> args{
      make_with_value<ArgumentTypes>(used_for_size, 0.0)...};
  // We fill with random values after initialization because the order of
  // evaluation is not guaranteed for a constructor call and so then knowing
  // the seed would not lead to reproducible results.
  (void)std::initializer_list<char>{(
      (void)fill_with_random_values(
          make_not_null(&std::get<ArgumentIs>(args)), make_not_null(&generator),
          make_not_null(&(distributions[ArgumentIs]))),
      '0')...};

  size_t count = 0;
  tmpl::for_each<TagsList>([&f, &klass, &args, &used_for_size, &member_args,
                            &epsilon, &module_name, &function_names,
                            &count](auto tag) {
    (void)member_args;    // Avoid compiler warning
    (void)used_for_size;  // Avoid compiler warning
    using Tag = tmpl::type_from<decltype(tag)>;
    const auto result =
        tuples::get<Tag>((klass.*f)(std::get<ArgumentIs>(args)...));
    INFO("function: " << function_names[count]);
    CHECK_ITERABLE_CUSTOM_APPROX(
        result,
        (pypp::call<std::decay_t<decltype(result)>>(
            module_name, function_names[count], std::get<ArgumentIs>(args)...,
            forward_to_pypp<std::decay_t<decltype(result)>>(
                std::get<MemberArgsIs>(member_args), used_for_size)...)),
        Approx::custom().epsilon(epsilon).scale(1.0));
    count++;
  });
}

template <class F, class T, class Klass, class... ArgumentTypes,
          class... MemberArgs, size_t... ArgumentIs, size_t... MemberArgsIs>
void check_with_random_values_impl(
    F&& f, const Klass& klass, const std::string& module_name,
    const std::string& function_name, std::mt19937 generator,
    std::array<std::uniform_real_distribution<>, sizeof...(ArgumentTypes)>
        distributions,
    const std::tuple<MemberArgs...>& member_args, const T& used_for_size,
    tmpl::list<ArgumentTypes...> /*argument_types*/,
    std::index_sequence<ArgumentIs...> /*index_argument_types*/,
    std::index_sequence<MemberArgsIs...> /*index_member_args*/,
    NoSuchType /*meta*/, const double epsilon = 1.0e-12) {
  // Note: generator and distributions cannot be const.
  using f_info = tt::function_info<cpp20::remove_cvref_t<F>>;
  using ResultType = typename f_info::return_type;
  std::tuple<ArgumentTypes...> args{
      make_with_value<ArgumentTypes>(used_for_size, 0.0)...};
  // We fill with random values after initialization because the order of
  // evaluation is not guaranteed for a constructor call and so then knowing the
  // seed would not lead to reproducible results.
  (void)std::initializer_list<char>{(
      (void)fill_with_random_values(
          make_not_null(&std::get<ArgumentIs>(args)), make_not_null(&generator),
          make_not_null(&(distributions[ArgumentIs]))),
      '0')...};
  const auto result = make_overloader(
      [&](std::true_type /*is_class*/, auto&& local_f) {
        return (klass.*local_f)(std::get<ArgumentIs>(args)...);
      },
      [&](std::false_type /*is_class*/, auto&& local_f) {
        return local_f(std::get<ArgumentIs>(args)...);
      })(
      std::integral_constant<
          bool, not cpp17::is_same_v<NoSuchType, std::decay_t<Klass>>>{},
      std::forward<F>(f));
  INFO("function: " << function_name);
  CHECK_ITERABLE_CUSTOM_APPROX(
      result,
      pypp::call<ResultType>(
          module_name, function_name, std::get<ArgumentIs>(args)...,
          forward_to_pypp<ResultType>(std::get<MemberArgsIs>(member_args),
                                      used_for_size)...),
      Approx::custom().epsilon(epsilon).scale(1.0));
}

template <class F, class T, class Klass, class... ReturnTypes,
          class... ArgumentTypes, class... MemberArgs, size_t... ResultIs,
          size_t... ArgumentIs, size_t... MemberArgsIs>
void check_with_random_values_impl(
    F&& f, const Klass& klass, const std::string& module_name,
    const std::vector<std::string>& function_names, std::mt19937 generator,
    std::array<std::uniform_real_distribution<>, sizeof...(ArgumentTypes)>
        distributions,
    const std::tuple<MemberArgs...>& member_args, const T& used_for_size,
    tmpl::list<ReturnTypes...> /*return_types*/,
    tmpl::list<ArgumentTypes...> /*argument_types*/,
    std::index_sequence<ResultIs...> /*index_return_types*/,
    std::index_sequence<ArgumentIs...> /*index_argument_types*/,
    std::index_sequence<MemberArgsIs...> /*index_member_args*/,
    NoSuchType /* meta */, const double epsilon = 1.0e-12) {
  // Note: generator and distributions cannot be const.
  std::tuple<ReturnTypes...> results{
      make_with_value<ReturnTypes>(used_for_size, 0.0)...};
  std::tuple<ArgumentTypes...> args{
      make_with_value<ArgumentTypes>(used_for_size, 0.0)...};
  // We fill with random values after initialization because the order of
  // evaluation is not guaranteed for a constructor call and so then knowing the
  // seed would not lead to reproducible results.
  (void)std::initializer_list<char>{(
      (void)fill_with_random_values(
          make_not_null(&std::get<ArgumentIs>(args)), make_not_null(&generator),
          make_not_null(&(distributions[ArgumentIs]))),
      '0')...};
  // We intentionally do not set the return value to signaling NaN so that not
  // all of our functions need to be able to handle the cases where they
  // receive a NaN. Instead, we fill the return value with random numbers.
  (void)std::initializer_list<char>{(
      (void)fill_with_random_values(make_not_null(&std::get<ResultIs>(results)),
                                    make_not_null(&generator),
                                    make_not_null(&(distributions[0]))),
      '0')...};
  make_overloader(
      [&](std::true_type /*is_class*/, auto&& local_f) {
        (klass.*local_f)(make_not_null(&std::get<ResultIs>(results))...,
                         std::get<ArgumentIs>(args)...);
      },
      [&](std::false_type /*is_class*/, auto&& local_f) {
        local_f(make_not_null(&std::get<ResultIs>(results))...,
                std::get<ArgumentIs>(args)...);
      })(
      std::integral_constant<
          bool, not cpp17::is_same_v<NoSuchType, std::decay_t<Klass>>>{},
      std::forward<F>(f));
  const auto helper = [&module_name, &function_names, &args, &results, &epsilon,
                       &member_args, &used_for_size](auto result_i) {
    (void)member_args;    // avoid compiler warning
    (void)used_for_size;  // avoid compiler warning
    constexpr size_t iter = decltype(result_i)::value;
    INFO("function: " << function_names[iter]);
    CHECK_ITERABLE_CUSTOM_APPROX(
        std::get<iter>(results),
        (pypp::call<std::tuple_element_t<iter, std::tuple<ReturnTypes...>>>(
            module_name, function_names[iter], std::get<ArgumentIs>(args)...,
            forward_to_pypp<
                std::tuple_element_t<iter, std::tuple<ReturnTypes...>>>(
                std::get<MemberArgsIs>(member_args), used_for_size)...)),
        Approx::custom().epsilon(epsilon).scale(1.0));
    return '0';
  };
  (void)std::initializer_list<char>{
      helper(std::integral_constant<size_t, ResultIs>{})...};
}
}  // namespace TestHelpers_detail

/*!
 * \brief Tests a C++ function returning by value by comparing the result to a
 * python function
 *
 * Tests the function `f` by comparing the result to that of the python function
 * `function_name` in the file `module_name`. The function is tested by
 * generated random values in the half-open range [`lower_bound`,
 * `upper_bound`). The argument `used_for_size` is used for constructing the
 * arguments of `f` by calling `make_with_value<ArgumentType>(used_for_size,
 * 0.0)`.
 *
 * \note You must explicitly pass the number of bounds you will be passing as
 * the first template parameter, the rest will be inferred.
 *
 * \note If you have a test fail you can replay the scenario by feeding in the
 * seed that was printed out in the failed test as the last argument.
 *
 * \param f The C++ function to test
 * \param module_name The python file relative to the directory used in
 * `SetupLocalPythonEnvironment`
 * \param function_name The name of the python function inside `module_name`
 * \param lower_and_upper_bounds The lower and upper bounds for the randomly
 * generated numbers. Must be either an array of a single pair, or of as many
 * pairs as there are arguments to `f` that are not a `gsl::not_null`
 * \param used_for_size The type `X` for the arguments of `f` of type
 *`Tensor<X>`
 * \param epsilon A double specifying the comparison tolerance
 * (default 1.0e-12)
 * \param seed The seed for the random number generator. This should only be
 * specified when debugging a failure with a particular set of random numbers,
 * in general it should be left to the default value.
 */
template <size_t NumberOfBounds, class F, class T,
          Requires<not cpp17::is_same_v<
              typename tt::function_info<cpp20::remove_cvref_t<F>>::return_type,
              void>> = nullptr>
// The Requires is used so that we can call the std::vector<std::string> with
// braces and not have it be ambiguous.
void check_with_random_values(
    F&& f, const std::string& module_name, const std::string& function_name,
    const std::array<std::pair<double, double>, NumberOfBounds>&
        lower_and_upper_bounds,
    const T& used_for_size, const double epsilon = 1.0e-12,
    const typename std::random_device::result_type seed =
        std::random_device{}()) {
  INFO("seed: " << seed);
  std::mt19937 generator(seed);
  using f_info = tt::function_info<cpp20::remove_cvref_t<F>>;
  using number_of_not_null =
      tmpl::count_if<typename f_info::argument_types,
                     tmpl::bind<TestHelpers_detail::is_not_null, tmpl::_1>>;
  using argument_types = tmpl::transform<
      tmpl::pop_front<typename f_info::argument_types, number_of_not_null>,
      std::decay<tmpl::_1>>;

  static_assert(number_of_not_null::value == 0,
                "Cannot return arguments by gsl::not_null if the python "
                "function name is passed as a string. If the function only "
                "returns one gsl::not_null then you must pass in a one element "
                "vector<string>.");
  static_assert(tmpl::size<argument_types>::value != 0,
                "The function 'f' must take at least one argument.");
  static_assert(NumberOfBounds == 1 or
                    NumberOfBounds == tmpl::size<argument_types>::value,
                "The number of lower and upper bound pairs must be either 1 or "
                "equal to the number of arguments taken by f that are not "
                "gsl::not_null.");
  std::array<std::uniform_real_distribution<>,
             tmpl::size<argument_types>::value>
      distributions;
  for (size_t i = 0; i < tmpl::size<argument_types>::value; ++i) {
    gsl::at(distributions, i) = std::uniform_real_distribution<>{
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).first,
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).second};
  }
  TestHelpers_detail::check_with_random_values_impl(
      std::forward<F>(f), NoSuchType{}, module_name, function_name, generator,
      std::move(distributions), std::tuple<>{}, used_for_size, argument_types{},
      std::make_index_sequence<tmpl::size<argument_types>::value>{},
      std::make_index_sequence<0>{}, NoSuchType{}, epsilon);
}

/*!
 * \brief Tests a C++ function returning by `gsl::not_null`  by comparing the
 * result to a python function
 *
 * Tests the function `f` by comparing the result to that of the python function
 * `function_name` in the file `module_name`. The function is tested by
 * generated random values in the half-open range [`lower_bound`, `upper_bound`)
 * for each argument. The argument `used_for_size` is used for constructing the
 * arguments of `f` by calling `make_with_value<ArgumentType>(used_for_size,
 * 0.0)`. For functions that return by `gsl::not_null`, the result will be
 * initialized with random values rather than to signaling `NaN`s. This means
 * functions do not need to support receiving a signaling `NaN` in their return
 * argument to be tested using this function.
 *
 * \note You must explicitly pass the number of bounds you will be passing as
 * the first template parameter, the rest will be inferred.
 *
 * \note If you have a test fail you can replay the scenario by feeding in the
 * seed that was printed out in the failed test as the last argument.
 *
 * \param f The C++ function to test
 * \param module_name The python file relative to the directory used in
 * `SetupLocalPythonEnvironment`
 * \param function_names The names of the python functions inside `module_name`
 * in the order that they return the `gsl::not_null` results
 * \param lower_and_upper_bounds The lower and upper bounds for the randomly
 * generated numbers. Must be either an array of a single pair, or of as many
 * pairs as there are arguments to `f` that are not a `gsl::not_null`
 * \param used_for_size The type `X` for the arguments of `f` of type
 * `Tensor<X>`
 * \param epsilon A double specifying the comparison tolerance
 * (default 1.0e-12)
 * \param seed The seed for the random number generator. This should only be
 * specified when debugging a failure with a particular set of random numbers,
 * in general it should be left to the default value.
 */
template <size_t NumberOfBounds, class F, class T>
void check_with_random_values(
    F&& f, const std::string& module_name,
    const std::vector<std::string>& function_names,
    const std::array<std::pair<double, double>, NumberOfBounds>&
        lower_and_upper_bounds,
    const T& used_for_size, const double epsilon = 1.0e-12,
    const typename std::random_device::result_type seed =
        std::random_device{}()) {
  INFO("seed: " << seed);
  std::mt19937 generator(seed);
  using f_info = tt::function_info<cpp20::remove_cvref_t<F>>;
  using number_of_not_null =
      tmpl::count_if<typename f_info::argument_types,
                     tmpl::bind<TestHelpers_detail::is_not_null, tmpl::_1>>;
  using argument_types = tmpl::transform<
      tmpl::pop_front<typename f_info::argument_types, number_of_not_null>,
      std::decay<tmpl::_1>>;
  using return_types =
      tmpl::transform<tmpl::pop_back<typename f_info::argument_types,
                                     tmpl::size<argument_types>>,
                      TestHelpers_detail::RemoveNotNull<tmpl::_1>>;

  static_assert(number_of_not_null::value != 0,
                "You must return at least one argument by gsl::not_null when "
                "passing the python function names as a vector<string>. If "
                "your function returns by value do not pass the function name "
                "as a vector<string> but just a string.");
  static_assert(
      cpp17::is_same_v<typename f_info::return_type, void>,
      "A function returning by gsl::not_null must have a void return type.");
  static_assert(tmpl::size<argument_types>::value != 0,
                "The function 'f' must take at least one argument.");
  static_assert(NumberOfBounds == 1 or
                    NumberOfBounds == tmpl::size<argument_types>::value,
                "The number of lower and upper bound pairs must be either 1 or "
                "equal to the number of arguments taken by f that are not "
                "gsl::not_null.");
  if (function_names.size() != number_of_not_null::value) {
    ERROR(
        "The number of python functions passed must be the same as the number "
        "of gsl::not_null arguments in the C++ function. The order of the "
        "python functions must also be the same as the order of the "
        "gsl::not_null arguments.");
  }
  std::array<std::uniform_real_distribution<>,
             tmpl::size<argument_types>::value>
      distributions;
  for (size_t i = 0; i < tmpl::size<argument_types>::value; ++i) {
    gsl::at(distributions, i) = std::uniform_real_distribution<>{
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).first,
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).second};
  }
  TestHelpers_detail::check_with_random_values_impl(
      std::forward<F>(f), NoSuchType{}, module_name, function_names, generator,
      std::move(distributions), std::tuple<>{}, used_for_size, return_types{},
      argument_types{},
      std::make_index_sequence<tmpl::size<return_types>::value>{},
      std::make_index_sequence<tmpl::size<argument_types>::value>{},
      std::make_index_sequence<0>{}, NoSuchType{}, epsilon);
}

/*!
 * \brief Tests a member function of a class returning by value by comparing the
 * result to a python function
 *
 * Tests the function `f` by comparing the result to that of the python function
 * `function_name` in the file `module_name`. An instance of the class is passed
 * in as the second argument and is the object on which the member function `f`
 * will be invoked. The member function is invoked as `klass.function`, so
 * passing in pointers is not supported. The function is tested by generated
 * random values in the half-open range [`lower_bound`, `upper_bound`). The
 * argument `used_for_size` is used for constructing the arguments of `f` by
 * calling `make_with_value<ArgumentType>(used_for_size, 0.0)`.
 *
 * \note You must explicitly pass the number of bounds you will be passing
 * as the first template parameter, the rest will be inferred.
 *
 * \note If you have a test fail you can replay the scenario by feeding in
 * the seed that was printed out in the failed test as the last argument.
 *
 * \param f The member function to test
 * \param klass the object on which to invoke `f`
 * \param module_name The python file relative to the directory used in
 * `SetupLocalPythonEnvironment`
 * \param function_name The name of the python function inside `module_name`
 * \param lower_and_upper_bounds The lower and upper bounds for the randomly
 * generated numbers. Must be either an array of a single pair, or of as many
 * pairs as there are arguments to `f` that are not a `gsl::not_null`
 * \param member_args a tuple of the member variables of the object `klass` that
 * the python function will need in order to perform the computation. These
 * should have the same types as the normal arguments passed to the member
 * function, e.g. `Tensor<X>`.
 * \param used_for_size The type `X` for the arguments of `f` of type
 * `Tensor<X>`
 * \param epsilon A double specifying the comparison tolerance
 * (default 1.0e-12)
 * \param seed The seed for the random number generator. This should only be
 * specified when debugging a failure with a particular set of random numbers,
 * in general it should be left to the default value.
 */
template <size_t NumberOfBounds, class F, class T, class... MemberArgs,
          Requires<not cpp17::is_same_v<
              typename tt::function_info<cpp20::remove_cvref_t<F>>::return_type,
              void>> = nullptr>
// The Requires is used so that we can call the std::vector<std::string> with
// braces and not have it be ambiguous.
void check_with_random_values(
    F&& f,
    const typename tt::function_info<cpp20::remove_cvref_t<F>>::class_type&
        klass,
    const std::string& module_name, const std::string& function_name,
    const std::array<std::pair<double, double>, NumberOfBounds>&
        lower_and_upper_bounds,
    const std::tuple<MemberArgs...>& member_args, const T& used_for_size,
    const double epsilon = 1.0e-12,
    const typename std::random_device::result_type seed =
        std::random_device{}()) {
  INFO("seed: " << seed);
  std::mt19937 generator(seed);
  using f_info = tt::function_info<cpp20::remove_cvref_t<F>>;
  using number_of_not_null =
      tmpl::count_if<typename f_info::argument_types,
                     tmpl::bind<TestHelpers_detail::is_not_null, tmpl::_1>>;
  using argument_types = tmpl::transform<
      tmpl::pop_front<typename f_info::argument_types, number_of_not_null>,
      std::decay<tmpl::_1>>;

  static_assert(number_of_not_null::value == 0,
                "Cannot return arguments by gsl::not_null if the python "
                "function name is passed as a string. If the function only "
                "returns one gsl::not_null then you must pass in a one element "
                "vector<string>.");
  static_assert(tmpl::size<argument_types>::value != 0,
                "The function 'f' must take at least one argument.");
  static_assert(NumberOfBounds == 1 or
                    NumberOfBounds == tmpl::size<argument_types>::value,
                "The number of lower and upper bound pairs must be either 1 or "
                "equal to the number of arguments taken by f that are not "
                "gsl::not_null.");
  std::array<std::uniform_real_distribution<>,
             tmpl::size<argument_types>::value>
      distributions;
  for (size_t i = 0; i < tmpl::size<argument_types>::value; ++i) {
    gsl::at(distributions, i) = std::uniform_real_distribution<>{
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).first,
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).second};
  }
  TestHelpers_detail::check_with_random_values_impl(
      std::forward<F>(f), klass, module_name, function_name, generator,
      std::move(distributions), member_args, used_for_size, argument_types{},
      std::make_index_sequence<tmpl::size<argument_types>::value>{},
      std::make_index_sequence<sizeof...(MemberArgs)>{}, NoSuchType{}, epsilon);
}

/*!
 * \brief Tests a member function of a class returning by either `gsl::not_null`
 * or `TaggedTuple` by comparing the result to a python function
 *
 * Tests the function `f` by comparing the result to that of the python
 * functions `function_names` in the file `module_name`. An instance of the
 * class is passed in as the second argument and is the object on which the
 * member function `f` will be invoked. The member function is invoked as
 * `klass.function`, so passing in pointers is not supported. The function is
 * tested by generated random values in the half-open range [`lower_bound`,
 * `upper_bound`). The argument `used_for_size` is used for constructing the
 * arguments of `f` by calling `make_with_value<ArgumentType>(used_for_size,
 * 0.0)`. For functions that return by `gsl::not_null`, the result will be
 * initialized with random values rather than to signaling `NaN`s. This means
 * functions do not need to support receiving a signaling `NaN` in their return
 * argument to be tested using this function.
 *
 * If `TagsList` is passed as a `tmpl::list`, then `f` is expected to
 * return a TaggedTuple. The result of each python function will be
 * compared with calling `tuples::get` on the result of `f`. The order of the
 * tags within `TagsList` should match the order of the functions in
 * `function_names`
 *
 * \note You must explicitly pass the number of bounds you will be passing as
 * the first template parameter, the rest will be inferred.
 *
 * \note If you have a test fail you can replay the scenario by feeding in the
 * seed that was printed out in the failed test as the last argument.
 *
 * \param f The member function to test
 * \param klass the object on which to invoke `f`
 * \param module_name The python file relative to the directory used in
 * `SetupLocalPythonEnvironment`
 * \param function_names The names of the python functions inside `module_name`
 * in the order that they return the `gsl::not_null` results
 * \param lower_and_upper_bounds The lower and upper bounds for the randomly
 * generated numbers. Must be either an array of a single pair, or of as many
 * pairs as there are arguments to `f` that are not a `gsl::not_null`
 * \param member_args a tuple of the member variables of the object `klass` that
 * the python function will need in order to perform the computation. These
 * should have the same types as the normal arguments passed to the member
 * function, e.g. `Tensor<X>`.
 * \param used_for_size The type `X` for the arguments of `f` of type
 * `Tensor<X>`
 * \param epsilon A double specifying the comparison tolerance
 * (default 1.0e-12)
 * \param seed The seed for the random number generator. This should only be
 * specified when debugging a failure with a particular set of random numbers,
 * in general it should be left to the default value.
 */
template <size_t NumberOfBounds, typename TagsList = NoSuchType, class F,
          class T, class... MemberArgs>
void check_with_random_values(
    F&& f,
    const typename tt::function_info<cpp20::remove_cvref_t<F>>::class_type&
        klass,
    const std::string& module_name,
    const std::vector<std::string>& function_names,
    const std::array<std::pair<double, double>, NumberOfBounds>&
        lower_and_upper_bounds,
    const std::tuple<MemberArgs...>& member_args, const T& used_for_size,
    const double epsilon = 1.0e-12,
    const typename std::random_device::result_type seed =
        std::random_device{}()) {
  INFO("seed: " << seed);
  std::mt19937 generator(seed);
  using f_info = tt::function_info<cpp20::remove_cvref_t<F>>;
  using number_of_not_null =
      tmpl::count_if<typename f_info::argument_types,
                     tmpl::bind<TestHelpers_detail::is_not_null, tmpl::_1>>;
  using argument_types = tmpl::transform<
      tmpl::pop_front<typename f_info::argument_types, number_of_not_null>,
      std::decay<tmpl::_1>>;
  using return_types =
      tmpl::transform<tmpl::pop_back<typename f_info::argument_types,
                                     tmpl::size<argument_types>>,
                      TestHelpers_detail::RemoveNotNull<tmpl::_1>>;

  static_assert(
      number_of_not_null::value != 0 or tt::is_a_v<tmpl::list, TagsList>,
      "You must either return at least one argument by gsl::not_null when"
      "passing the python function names as a vector<string>, or return by "
      "value using a TaggedTuple. If your function returns by value with a "
      "type that is not a TaggedTuple do not pass the function name as a "
      "vector<string> but just a string.");
  static_assert(cpp17::is_same_v<typename f_info::return_type, void> or
                    tt::is_a_v<tmpl::list, TagsList>,
                "The function must either return by gsl::not_null and have a "
                "void return type, or return by TaggedTuple");
  static_assert(tmpl::size<argument_types>::value != 0,
                "The function 'f' must take at least one argument.");
  static_assert(NumberOfBounds == 1 or
                    NumberOfBounds == tmpl::size<argument_types>::value,
                "The number of lower and upper bound pairs must be either 1 or "
                "equal to the number of arguments taken by f that are not "
                "gsl::not_null.");
  if (function_names.size() != number_of_not_null::value and
      not tt::is_a_v<tmpl::list, TagsList>) {
    ERROR(
        "If testing a function that returns by gsl::not_null, the number of "
        "python functions passed must be the same as the number of "
        "gsl::not_null arguments in the C++ function. The order of the "
        "python functions must also be the same as the order of the "
        "gsl::not_null arguments.");
  }
  std::array<std::uniform_real_distribution<>,
             tmpl::size<argument_types>::value>
      distributions;
  for (size_t i = 0; i < tmpl::size<argument_types>::value; ++i) {
    gsl::at(distributions, i) = std::uniform_real_distribution<>{
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).first,
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).second};
  }
  TestHelpers_detail::check_with_random_values_impl(
      std::forward<F>(f), klass, module_name, function_names, generator,
      std::move(distributions), member_args, used_for_size, return_types{},
      argument_types{},
      std::make_index_sequence<tmpl::size<return_types>::value>{},
      std::make_index_sequence<tmpl::size<argument_types>::value>{},
      std::make_index_sequence<sizeof...(MemberArgs)>{}, TagsList{}, epsilon);
}

/// \cond
#define INVOKE_FUNCTION_TUPLE_PUSH_BACK(r, DATA, ELEM) \
  BOOST_PP_TUPLE_PUSH_BACK(DATA, ELEM)

#define INVOKE_FUNCTION_WITH_MANY_TEMPLATE_PARAMS_IMPL(r, DATA)              \
  BOOST_PP_TUPLE_ELEM(                                                       \
      0, BOOST_PP_TUPLE_ELEM(                                                \
             0, DATA))<BOOST_PP_TUPLE_ELEM(2, BOOST_PP_TUPLE_ELEM(0, DATA)), \
                       BOOST_PP_TUPLE_ENUM(BOOST_PP_TUPLE_POP_FRONT(DATA))>  \
      BOOST_PP_TUPLE_ELEM(1, BOOST_PP_TUPLE_ELEM(0, DATA));

// The case where there is more than one tuple of template parameters to tensor
// product together. The first tuple of template parameters is transformed to a
// tuple of (FUNCTION_NAME, TUPLE_ARGS, TEMPLATE_PARAM). Then the macro will
// extract the values. The reason for needing to do this is that
// BOOST_PP_LIST_FOR_EACH_PRODUCT does not allow for passing extra data along
// to the macro.
#define INVOKE_FUNCTION_WITH_MANY_TEMPLATE_PARAMS(TUPLE_OF_TEMPLATE_PARAMS) \
  BOOST_PP_LIST_FOR_EACH_PRODUCT(                                           \
      INVOKE_FUNCTION_WITH_MANY_TEMPLATE_PARAMS_IMPL,                       \
      BOOST_PP_TUPLE_SIZE(TUPLE_OF_TEMPLATE_PARAMS), TUPLE_OF_TEMPLATE_PARAMS)

// The case where there is one tuple of template parameters to iterate over.
#define INVOKE_FUNCTION_WITH_SINGLE_TEMPLATE_PARAM(_, DATA, ELEM) \
  BOOST_PP_TUPLE_ELEM(0, DATA)<ELEM> BOOST_PP_TUPLE_ELEM(1, DATA);

#define INVOKE_FUNCTION_WITH_MANY_TEMPLATE_PARAMS_TUPLE_TO_LIST(r, _, ELEM) \
  BOOST_PP_TUPLE_TO_LIST(ELEM)
/// \endcond

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Macro used to invoke a test function of multiple template arguments.
 *
 * This macro allows to generate calls to multiple instances of
 * a test function template, all of which will receive the same parameters.
 * The first argument to this macro is the name of the function. The second
 * argument is a macro-tuple containing the parameters passed to each instance,
 * e.g. `(x, y)`. The remaining arguments are macro-tuples of the values for
 * each template parameter one wants to loop over, e.g.
 * `(1, 2, 3), (Frame::Inertial, Frame::Grid)`. For example, a function template
 *
 * \code
 * template <class Arg1, size_t Arg2, class Arg3>
 * my_function(const double& var_1, const int& var_2) noexcept { ... }
 * \endcode
 *
 * can be invoked by writing
 *
 * \code
 * INVOKE_TEST_FUNCTION(my_function, (d, i), (a, b, c), (1, 2, 3), (A, B, C))
 * \endcode
 *
 * which will generate
 *
 * \code
 * my_function<a, 1, A>(d, i);
 * my_function<a, 1, B>(d, i);
 * my_function<a, 1, C>(d, i);
 * my_function<a, 2, A>(d, i);
 * my_function<a, 2, B>(d, i);
 * my_function<a, 2, C>(d, i);
 * my_function<a, 3, A>(d, i);
 * my_function<a, 3, B>(d, i);
 * my_function<a, 3, C>(d, i);
 * my_function<b, 1, A>(d, i);
 * my_function<b, 1, B>(d, i);
 * my_function<b, 1, C>(d, i);
 * my_function<b, 2, A>(d, i);
 * my_function<b, 2, B>(d, i);
 * my_function<b, 2, C>(d, i);
 * my_function<b, 3, A>(d, i);
 * my_function<b, 3, B>(d, i);
 * my_function<b, 3, C>(d, i);
 * my_function<c, 1, A>(d, i);
 * my_function<c, 1, B>(d, i);
 * my_function<c, 1, C>(d, i);
 * my_function<c, 2, A>(d, i);
 * my_function<c, 2, B>(d, i);
 * my_function<c, 2, C>(d, i);
 * my_function<c, 3, A>(d, i);
 * my_function<c, 3, B>(d, i);
 * my_function<c, 3, C>(d, i);
 * \endcode
 *
 * \note The order of the macro-tuples of values must match the order of the
 * template parameters of the function.
 *
 * \note The function to be called must at least have one template argument,
 * so passing an empty set of template parameters will not work.
 */
#define INVOKE_TEST_FUNCTION(FUNCTION_NAME, TUPLE_ARGS, ...)                   \
  BOOST_PP_ASSERT_MSG(BOOST_PP_NOT(BOOST_VMD_IS_EMPTY(__VA_ARGS__)),           \
                      "You cannot pass an empty set of template parameters "   \
                      "to INVOKE_TEST_FUNCTION")                               \
  BOOST_PP_TUPLE_ENUM(                                                         \
      0,                                                                       \
      BOOST_PP_IF(                                                             \
          BOOST_PP_EQUAL(                                                      \
              BOOST_PP_TUPLE_SIZE(BOOST_PP_VARIADIC_TO_TUPLE(__VA_ARGS__)),    \
              1),                                                              \
          (BOOST_PP_LIST_FOR_EACH(                                             \
              INVOKE_FUNCTION_WITH_SINGLE_TEMPLATE_PARAM,                      \
              (FUNCTION_NAME, TUPLE_ARGS),                                     \
              BOOST_PP_TUPLE_TO_LIST(                                          \
                  BOOST_PP_VARIADIC_ELEM(0, __VA_ARGS__)))),                   \
          (INVOKE_FUNCTION_WITH_MANY_TEMPLATE_PARAMS(                          \
              BOOST_PP_TUPLE_PUSH_FRONT(                                       \
                  BOOST_PP_LIST_TO_TUPLE(BOOST_PP_LIST_TRANSFORM(              \
                      INVOKE_FUNCTION_WITH_MANY_TEMPLATE_PARAMS_TUPLE_TO_LIST, \
                      _,                                                       \
                      BOOST_PP_LIST_REST(                                      \
                          BOOST_PP_VARIADIC_TO_LIST(__VA_ARGS__)))),           \
                  BOOST_PP_LIST_TRANSFORM(                                     \
                      INVOKE_FUNCTION_TUPLE_PUSH_BACK,                         \
                      (FUNCTION_NAME, TUPLE_ARGS),                             \
                      BOOST_PP_TUPLE_TO_LIST(                                  \
                          BOOST_PP_VARIADIC_ELEM(0, __VA_ARGS__))))))))

/// \cond
#define GENERATE_UNINITIALIZED_DOUBLE \
  const double d(std::numeric_limits<double>::signaling_NaN())

#define GENERATE_UNINITIALIZED_DATAVECTOR const DataVector dv(5)

#define GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR \
  GENERATE_UNINITIALIZED_DOUBLE;                     \
  GENERATE_UNINITIALIZED_DATAVECTOR

#define CHECK_FOR_DOUBLES(FUNCTION_NAME, ...) \
  INVOKE_TEST_FUNCTION(FUNCTION_NAME, (d), __VA_ARGS__)

#define CHECK_FOR_DATAVECTORS(FUNCTION_NAME, ...) \
  INVOKE_TEST_FUNCTION(FUNCTION_NAME, (dv), __VA_ARGS__)
/// \endcond

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Macro used to test functions whose parameter can be a `double` or a
 * `DataVector`.
 *
 * In testing multiple instances of a function template using random values, it
 * often proves useful to write a wrapper around
 * `pypp::check_with_random_values`. This way, one can easily loop over several
 * values of one or multiple template parameters (e.g. when testing a
 * function templated in the number of spacetime dimensions.) The template
 * parameters of the wrapper will then correspond to the template parameters of
 * the function, which will be used by `pypp::check_with_random_values`
 * to invoke and test each instance. Each of these wrappers will generally
 * need only one parameter, namely a variable `used_for_size` passed to
 * `pypp::check_with_random_values` that can be a `double`, a `DataVector`, or
 * both (provided that the function being tested is templated in the type of
 * `used_for_size`.) Since this is applied in multiple test files, all of these
 * files will share the same way to generate the required calls to the wrapper.
 *
 * This macro, along with
 *
 * \code
 * CHECK_FOR_DOUBLES(FUNCTION_NAME, ...)
 * \endcode
 * \code
 * CHECK_FOR_DATAVECTORS(FUNCTION_NAME, ...)
 * \endcode
 *
 * allow to generate calls to multiple instances of a test function template in
 * the same way as done by `INVOKE_TEST_FUNCTION(FUNCTION_NAME, ARGS_TUPLE,
 * ...)`
 * (to which these macros call), except that the tuple of arguments is not
 * passed, as these macros will assume that a `double` `d`
 * and/or a `DataVector` `dv` will be previously defined. Although any `d`s and
 * `dv`s will work, one can (and it is recommended to) generate signaling `NaN`
 * values for `d` and `dv`. This can be done by invoking one of the three
 * provided macros: `GENERATE_UNINIATILIZED_DOUBLE`,
 * `GENERATE_UNINITIALIZED_DATAVECTOR`, or
 * `GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR`. For example,
 *
 * \code
 * GENERATE_UNINITIALIZED_DATAVECTOR;
 * CHECK_FOR_DATAVECTORS(test_fluxes, (1, 2, 3))
 * \endcode
 *
 * will generate a test case for 1, 2 and 3 dimensions:
 *
 * \code
 * const DataVector dv(5);
 * test_fluxes<1>(dv);
 * test_fluxes<2>(dv);
 * test_fluxes<3>(dv);
 * \endcode
 *
 * Analogously, the wrapper
 *
 * \code
 * template <size_t Dim, IndexType TypeOfIndex, typename DataType>
 * test_ricci(const DataType& used_for_size) noexcept { ... }
 * \endcode
 *
 * can be invoked by writing
 *
 * \code
 * GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
 *
 * CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_ricci, (1, 2, 3),
 *                                   (IndexType::Spatial, IndexType::Spacetime))
 * \endcode
 *
 * which will generate
 *
 * \code
 * const double d(std::numeric_limits<double>::signaling_NaN());
 * const DataVector dv(5);
 *
 * test_ricci<1, IndexType::Spatial>(d);
 * test_ricci<1, IndexType::Spacetime>(d);
 * test_ricci<2, IndexType::Spatial>(d);
 * test_ricci<2, IndexType::Spacetime>(d);
 * test_ricci<3, IndexType::Spatial>(d);
 * test_ricci<3, IndexType::Spacetime>(d);
 * test_ricci<1, IndexType::Spatial>(dv);
 * test_ricci<1, IndexType::Spacetime>(dv);
 * test_ricci<2, IndexType::Spatial>(dv);
 * test_ricci<2, IndexType::Spacetime>(dv);
 * test_ricci<3, IndexType::Spatial>(dv);
 * test_ricci<3, IndexType::Spacetime>(dv);
 * \endcode
 *
 * Note that it is not necessary to pass values for `DataType`, as they are
 * deduced from `used_for_size`.
 *
 * \note The order of the macro-tuples of values must match the order of the
 * template parameters of the function.
 *
 * \note The function to be called must at least have one template argument,
 * so passing an empty set of template parameters will not work.
 */
#define CHECK_FOR_DOUBLES_AND_DATAVECTORS(FUNCTION_NAME, ...) \
  CHECK_FOR_DOUBLES(FUNCTION_NAME, __VA_ARGS__)               \
  CHECK_FOR_DATAVECTORS(FUNCTION_NAME, __VA_ARGS__)
}  // namespace pypp
