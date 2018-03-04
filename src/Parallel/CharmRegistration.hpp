// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

#include "Parallel/ParallelComponentHelpers.hpp"

namespace Parallel {
namespace charmxx {
/// \cond
struct RegistrationHelper;
extern std::unique_ptr<RegistrationHelper>* charm_register_list;
extern size_t charm_register_list_capacity;
extern size_t charm_register_list_size;
/// \endcond

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Returns the template parameter as a `std::string`
 *
 * Uses the __PRETTY_FUNCTION__ compiler intrinsic to extract the template
 * parameter names in the same form that Charm++ uses to register entry methods.
 * This is used by the generated Singleton, Array, Group and Nodegroup headers,
 * as well as in CharmMain.cpp.
 */
template <class... Args>
std::string get_template_parameters_as_string() {
  std::string function_name(static_cast<char const*>(__PRETTY_FUNCTION__));
  std::string template_params =
      function_name.substr(function_name.find(std::string("Args = ")) + 8);
  template_params.erase(template_params.end() - 2, template_params.end());
  size_t pos = 0;
  while ((pos = template_params.find(" >")) != std::string::npos) {
    template_params.replace(pos, 1, ">");
    template_params.erase(pos + 1, 1);
  }
  pos = 0;
  while ((pos = template_params.find(", ", pos)) != std::string::npos) {
    template_params.erase(pos + 1, 1);
  }
  pos = 0;
  while ((pos = template_params.find('>', pos + 2)) != std::string::npos) {
    template_params.replace(pos, 1, " >");
  }
  std::replace(template_params.begin(), template_params.end(), '%', '>');
  // GCC's __PRETTY_FUNCTION__ adds the return type at the end, so we remove it.
  if (template_params.find('}') != std::string::npos) {
    template_params.erase(template_params.find('}'), template_params.size());
  }
  return template_params;
}

/*!
 * \ingroup CharmExtensionsGroup
 * \brief The base class used for automatic registration of entry methods.
 *
 * Entry methods are automatically registered by building a list of the entry
 * methods that need to be registered in the `charm_register_list`. All entry
 * methods in the list are later registered in the
 * `register_parallel_components` function, at which point the list is also
 * deleted.
 *
 * The reason for using an abstract base class mechanism is that we need to be
 * able to register entry method templates. The derived classes keep track of
 * the template parameters and override the `register_with_charm` function.
 * The result is that there must be one derived class template for each entry
 * method template, but since we only have a few entry method templates this is
 * not an issue.
 */
struct RegistrationHelper {
  RegistrationHelper() = default;
  RegistrationHelper(const RegistrationHelper&) = default;
  RegistrationHelper& operator=(const RegistrationHelper&) = default;
  RegistrationHelper(RegistrationHelper&&) = default;
  RegistrationHelper& operator=(RegistrationHelper&&) = default;
  virtual ~RegistrationHelper() = default;

  virtual void register_with_charm() const noexcept = 0;
  virtual std::string name() const noexcept = 0;
};

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Derived class for registering simple actions.
 *
 * Calls the appropriate Charm++ function to register a simple action.
 */
template <typename ParallelComponent, typename Action, typename... Args>
struct RegisterSimpleAction : RegistrationHelper {
  using chare_type = typename ParallelComponent::chare_type;
  using charm_type = charm_types_with_parameters<
      ParallelComponent, typename ParallelComponent::metavariables,
      typename ParallelComponent::action_list,
      typename get_array_index<chare_type>::template f<ParallelComponent>,
      typename ParallelComponent::initial_databox>;
  using cproxy = typename charm_type::cproxy;
  using ckindex = typename charm_type::ckindex;
  using algorithm = typename charm_type::algorithm;

  RegisterSimpleAction() = default;
  RegisterSimpleAction(const RegisterSimpleAction&) = default;
  RegisterSimpleAction& operator=(const RegisterSimpleAction&) = default;
  RegisterSimpleAction(RegisterSimpleAction&&) = default;
  RegisterSimpleAction& operator=(RegisterSimpleAction&&) = default;
  ~RegisterSimpleAction() override = default;

  void register_with_charm() const noexcept override {
    static bool done_registration{false};
    if (done_registration) {
      return;  // LCOV_EXCL_LINE
    }
    done_registration = true;
    ckindex::template idx_explicit_single_action<Action>(
        static_cast<void (algorithm::*)(const std::tuple<Args...>&)>(nullptr));
  }

  std::string name() const noexcept override {
    return get_template_parameters_as_string<RegisterSimpleAction>();
  }

  static bool registrar;
};

/// \cond
template <typename ParallelComponent, typename Action>
struct RegisterSimpleAction<ParallelComponent, Action> : RegistrationHelper {
  using chare_type = typename ParallelComponent::chare_type;
  using charm_type = charm_types_with_parameters<
      ParallelComponent, typename ParallelComponent::metavariables,
      typename ParallelComponent::action_list,
      typename get_array_index<chare_type>::template f<ParallelComponent>,
      typename ParallelComponent::initial_databox>;
  using cproxy = typename charm_type::cproxy;
  using ckindex = typename charm_type::ckindex;
  using algorithm = typename charm_type::algorithm;

  RegisterSimpleAction() = default;
  RegisterSimpleAction(const RegisterSimpleAction&) = default;
  RegisterSimpleAction& operator=(const RegisterSimpleAction&) = default;
  RegisterSimpleAction(RegisterSimpleAction&&) = default;
  RegisterSimpleAction& operator=(RegisterSimpleAction&&) = default;
  ~RegisterSimpleAction() override = default;

  void register_with_charm() const noexcept override {
    static bool done_registration{false};
    if (done_registration) {
      return;  // LCOV_EXCL_LINE
    }
    done_registration = true;
    ckindex::template idx_explicit_single_action<Action>(
        static_cast<void (algorithm::*)()>(nullptr));
  }

  std::string name() const noexcept override {
    return get_template_parameters_as_string<RegisterSimpleAction>();
  }

  static bool registrar;
};
/// \endcond

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Function that adds a pointer to a specific derived class to the
 * `charm_register_list`
 *
 * Used to initialize the `registrar` bool of derived classes of
 * `RegistrationHelper`. When the function is invoked it appends the derived
 * class to the `charm_register_list`.
 *
 * \note The reason for not using a `std::vector` is that this did not behave
 * correctly when calling `push_back`. Specifically, the final vector was always
 * size 1, even though multiple elements were pushed back. The reason for that
 * behavior was never tracked down and so in the future it could be possible to
 * use a `std::vector`.
 */
template <typename Derived>
bool register_func_with_charm() noexcept {
  if (charm_register_list_size >= charm_register_list_capacity) {
    auto* const t =
        new std::unique_ptr<RegistrationHelper>[charm_register_list_capacity +
                                                10];
    for (size_t i = 0; i < charm_register_list_capacity; ++i) {
      // clang-tidy: do not use pointer arithmetic
      t[i] = std::move(charm_register_list[i]);  // NOLINT
    }
    delete[] charm_register_list;
    charm_register_list = t;
    charm_register_list_capacity += 10;
  }
  charm_register_list_size++;
  // clang-tidy: do not use pointer arithmetic
  charm_register_list[charm_register_list_size - 1] =  // NOLINT
      std::make_unique<Derived>();
  return true;
}
}  // namespace charmxx
}  // namespace Parallel

// clang-tidy: redundant declaration
template <typename ParallelComponent, typename Action, typename... Args>
bool Parallel::charmxx::RegisterSimpleAction<ParallelComponent, Action,
                                             Args...>::registrar =  // NOLINT
    Parallel::charmxx::register_func_with_charm<
        RegisterSimpleAction<ParallelComponent, Action, Args...>>();

// clang-tidy: redundant declaration
template <typename ParallelComponent, typename Action>
bool Parallel::charmxx::RegisterSimpleAction<ParallelComponent,
                                             Action>::registrar =  // NOLINT
    Parallel::charmxx::register_func_with_charm<
        RegisterSimpleAction<ParallelComponent, Action>>();

/// \cond
class CkReductionMsg;
/// \endcond

namespace Parallel {
namespace charmxx {
/*!
 * \ingroup CharmExtensionsGroup
 * \brief The type of a function pointer to a Charm++ custom reduction function.
 */
using ReducerFunctions = CkReductionMsg* (*)(int, CkReductionMsg**);
/// \cond
extern ReducerFunctions* charm_reducer_functions_list;
extern size_t charm_reducer_functions_capacity;
extern size_t charm_reducer_functions_size;
extern std::unordered_map<size_t, CkReduction::reducerType>
    charm_reducer_functions;
/// \endcond

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Class used for registering custom reduction function
 *
 * The static member variable is initialized before main is entered. This means
 * we are able to inject code into the beginning of the execution of an
 * executable by "using" the variable (casting to void counts) the function that
 * initializes the variable is called. We "use" `registrar` inside the
 * `Parallel::contribute_to_reduction` function.
 */
template <ReducerFunctions>
struct RegisterReducerFunction {
  static bool registrar;
};

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Function that stores a function pointer to the custom reduction
 * function to be registered later.
 *
 * Used to initialize the `registrar` bool of `RegisterReducerFunction`. When
 * invoked it adds the function `F` of type `ReducerFunctions` to the list
 * `charm_reducer_functions_list`.
 *
 * \note The reason for not using a `std::vector<ReducerFunctions>` is that this
 * did not behave correctly when calling `push_back`. Specifically, the final
 * vector was always size 1, even though multiple elements were pushed back. The
 * reason for that behavior was never tracked down and so in the future it could
 * be possible to use a `std::vector`.
 */
template <ReducerFunctions F>
bool register_reducer_function() noexcept {
  if (charm_reducer_functions_size >= charm_reducer_functions_capacity) {
    auto* const t = new ReducerFunctions[charm_reducer_functions_capacity + 10];
    for (size_t i = 0; i < charm_reducer_functions_capacity; ++i) {
      // clang-tidy: do not use pointer arithmetic
      t[i] = std::move(charm_reducer_functions_list[i]);  // NOLINT
    }
    delete[] charm_reducer_functions_list;
    charm_reducer_functions_list = t;
    charm_reducer_functions_capacity += 10;
  }
  charm_reducer_functions_size++;
  // clang-tidy: do not use pointer arithmetic
  charm_reducer_functions_list[charm_reducer_functions_size - 1] = F;  // NOLINT
  return true;
}
}  // namespace charmxx
}  // namespace Parallel

// clang-tidy: do not use pointer arithmetic
template <Parallel::charmxx::ReducerFunctions F>
bool Parallel::charmxx::RegisterReducerFunction<F>::registrar =  // NOLINT
    Parallel::charmxx::register_reducer_function<F>();
