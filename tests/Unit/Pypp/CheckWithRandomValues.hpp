// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <initializer_list>
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
    TagsList /*meta*/) {
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
                            &module_name, &function_names, &count](auto tag) {
    (void)used_for_size;  // Avoid compiler warning
    using Tag = tmpl::type_from<decltype(tag)>;
    const auto result =
        tuples::get<Tag>((klass.*f)(std::get<ArgumentIs>(args)...));
    CHECK_ITERABLE_APPROX(
        result,
        (pypp::call<std::decay_t<decltype(result)>>(
            module_name, function_names[count], std::get<ArgumentIs>(args)...,
            forward_to_pypp<std::decay_t<decltype(result)>>(
                std::get<MemberArgsIs>(member_args), used_for_size)...)));
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
    NoSuchType /*meta*/) {
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
  CHECK_ITERABLE_APPROX(
      result, pypp::call<ResultType>(
                  module_name, function_name, std::get<ArgumentIs>(args)...,
                  forward_to_pypp<ResultType>(
                      std::get<MemberArgsIs>(member_args), used_for_size)...));
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
    NoSuchType /* meta */) {
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
  const auto helper = [&module_name, &function_names, &args, &results,
                       &member_args, &used_for_size](auto result_i) {
    (void)member_args;  // avoid compiler warning
    (void)used_for_size;  // avoid compiler warning
    constexpr size_t iter = decltype(result_i)::value;
    CHECK_ITERABLE_APPROX(
        std::get<iter>(results),
        (pypp::call<std::tuple_element_t<iter, std::tuple<ReturnTypes...>>>(
            module_name, function_names[iter], std::get<ArgumentIs>(args)...,
            forward_to_pypp<
                std::tuple_element_t<iter, std::tuple<ReturnTypes...>>>(
                std::get<MemberArgsIs>(member_args), used_for_size)...)));
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
    const T& used_for_size,
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
    gsl::at(distributions,
            NumberOfBounds == 1 ? 0 : i) = std::uniform_real_distribution<>{
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).first,
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).second};
  }
  TestHelpers_detail::check_with_random_values_impl(
      std::forward<F>(f), NoSuchType{}, module_name, function_name, generator,
      std::move(distributions), std::tuple<>{}, used_for_size, argument_types{},
      std::make_index_sequence<tmpl::size<argument_types>::value>{},
      std::make_index_sequence<0>{}, NoSuchType{});
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
    const T& used_for_size,
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
    gsl::at(distributions,
            NumberOfBounds == 1 ? 0 : i) = std::uniform_real_distribution<>{
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).first,
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).second};
  }
  TestHelpers_detail::check_with_random_values_impl(
      std::forward<F>(f), NoSuchType{}, module_name, function_names, generator,
      std::move(distributions), std::tuple<>{}, used_for_size, return_types{},
      argument_types{},
      std::make_index_sequence<tmpl::size<return_types>::value>{},
      std::make_index_sequence<tmpl::size<argument_types>::value>{},
      std::make_index_sequence<0>{}, NoSuchType{});
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
    gsl::at(distributions,
            NumberOfBounds == 1 ? 0 : i) = std::uniform_real_distribution<>{
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).first,
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).second};
  }
  TestHelpers_detail::check_with_random_values_impl(
      std::forward<F>(f), klass, module_name, function_name, generator,
      std::move(distributions), member_args, used_for_size, argument_types{},
      std::make_index_sequence<tmpl::size<argument_types>::value>{},
      std::make_index_sequence<sizeof...(MemberArgs)>{}, NoSuchType{});
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
    gsl::at(distributions,
            NumberOfBounds == 1 ? 0 : i) = std::uniform_real_distribution<>{
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).first,
        gsl::at(lower_and_upper_bounds, NumberOfBounds == 1 ? 0 : i).second};
  }
  TestHelpers_detail::check_with_random_values_impl(
      std::forward<F>(f), klass, module_name, function_names, generator,
      std::move(distributions), member_args, used_for_size, return_types{},
      argument_types{},
      std::make_index_sequence<tmpl::size<return_types>::value>{},
      std::make_index_sequence<tmpl::size<argument_types>::value>{},
      std::make_index_sequence<sizeof...(MemberArgs)>{}, TagsList{});
}
}  // namespace pypp
