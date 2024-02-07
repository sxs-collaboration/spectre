// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions used by Charm++ to register classes/chares and entry
/// methods

#pragma once

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

#include "Parallel/CharmRegistration.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/NoSuchType.hpp"

namespace Parallel::charmxx {
/// \cond
std::unique_ptr<RegistrationHelper>* charm_register_list = nullptr;
size_t charm_register_list_capacity = 0;
size_t charm_register_list_size = 0;

ReducerFunctions* charm_reducer_functions_list = nullptr;
size_t charm_reducer_functions_capacity = 0;
size_t charm_reducer_functions_size = 0;
std::unordered_map<size_t, CkReduction::reducerType> charm_reducer_functions{};
/// \endcond

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Registers parallel components and their entry methods with Charm++
 */
inline void register_parallel_components() {
  static bool done_registration{false};
  if (done_registration) {
    return;  // LCOV_EXCL_LINE
  }
  done_registration = true;

  Parallel::PrinterChare::register_with_charm();

  // Charm++ requires the order of registration to be the same across all
  // processors. To make sure this is satisfied regardless of any possible weird
  // behavior with static variable initialization we sort the list before
  // registering anything.
  // clang-tidy: do not user pointer arithmetic
  std::sort(charm_register_list,
            charm_register_list + charm_register_list_size,  // NOLINT
            [](const std::unique_ptr<RegistrationHelper>& a,
               const std::unique_ptr<RegistrationHelper>& b) {
              return a->name() < b->name();
            });

  // register chares and parallel components
  for (size_t i = 0;
       i < charm_register_list_size and i < charm_register_list_capacity; ++i) {
    // clang-tidy: do not use pointer arithmetic
    if (charm_register_list[i]->is_registering_chare()) {  // NOLINT
      charm_register_list[i]->register_with_charm();       // NOLINT
    }
  }
  // register reduction and simple actions, as well as receive_data
  for (size_t i = 0;
       i < charm_register_list_size and i < charm_register_list_capacity; ++i) {
    // clang-tidy: do not use pointer arithmetic
    if (not charm_register_list[i]->is_registering_chare()) {  // NOLINT
      charm_register_list[i]->register_with_charm();           // NOLINT
    }
  }
  // clang-tidy: use gsl::owner (we don't use raw owning pointers unless
  // necessary)
  delete[] charm_register_list;  // NOLINT
}

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Takes `charm_reducer_functions_list` and creates an unordered_map from
 * hashes of the functions to `CkReduction::reducerType`, which is used when
 * calling custom reductions.
 *
 * \note The reason `delete[]` isn't call here (or at all on
 * `charm_reducer_functions_list`) is because it has not been tested whether
 * that is safe to do in non-SMP builds.
 *
 * \see Parallel::contribute_to_reduction()
 */
inline void register_custom_reducer_functions() {
  for (size_t i = 0; i < charm_reducer_functions_size; ++i) {
    // clang-tidy: do not use pointer arithmetic
    charm_reducer_functions.emplace(
        std::hash<Parallel::charmxx::ReducerFunctions>{}(
            charm_reducer_functions_list[i]),                        // NOLINT
        CkReduction::addReducer(*charm_reducer_functions_list[i]));  // NOLINT
  }
}

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Register all init_node and init_proc functions with Charm++
 *
 * The `charm_init_node_funcs` are executed at startup by Charm++ on each node,
 * and the `charm_init_proc_funcs` on each processing element (PE). In addition
 * to these, some default init_node and init_proc functions are registered for
 * things like error handling.
 *
 * \note The function that registers the custom reducers is the first init_node
 * function registered.
 */
inline void register_init_node_and_proc(
    const std::vector<void (*)()>& charm_init_node_funcs,
    const std::vector<void (*)()>& charm_init_proc_funcs) {
  static bool done_registration{false};
  if (done_registration) {
    return;  // LCOV_EXCL_LINE
  }
  done_registration = true;
  // We explicitly register custom reducer functions first.
  _registerInitCall(register_custom_reducer_functions, 1);
  for (const auto& init_node_func :
       {&setup_error_handling, &setup_memory_allocation_failure_reporting,
        &disable_openblas_multithreading}) {
    _registerInitCall(*init_node_func, 1);
  }
  for (const auto& init_node_func : charm_init_node_funcs) {
    _registerInitCall(*init_node_func, 1);
  }
  for (const auto& init_proc_func :
       {&enable_floating_point_exceptions, &enable_segfault_handler}) {
    _registerInitCall(*init_proc_func, 0);
  }

  for (const auto& init_proc_func : charm_init_proc_funcs) {
    _registerInitCall(*init_proc_func, 0);
  }
}

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Register the Charm++ mainchare, reducers, chares, and entry methods.
 *
 * Call this function in `CkRegisterMainModule()` to set up a Charm++ executable
 * defined by the `Metavariables`. Normally the `CkRegisterMainModule()` is
 * generated by the `charmc` parser while generating
 * header files from interface files. The layer we built on Charm++ does not use
 * interface files directly and so we need to supply our `CkRegisterMainModule`
 * function. This function is also responsible for registering all parallel
 * components, all entry methods, and all custom reduction functions.
 */
template <typename Metavariables>
void register_main_module() {
  using charmxx_main_component = Parallel::Main<Metavariables>;
  (void)charmxx_main_component{
      Parallel::charmxx::MainChareRegistrationConstructor{}};
  Parallel::charmxx::register_parallel_components();
}

}  // namespace Parallel::charmxx
