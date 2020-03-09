// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <functional>

#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

namespace {
class A;
}  // namespace

/// [remove_reference_wrapper_example]
static_assert(
    cpp17::is_same_v<const double, tt::remove_reference_wrapper_t<
                                       std::reference_wrapper<const double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<const double,
                               tt::remove_reference_wrapper_t<const double>>,
              "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<
        double, tt::remove_reference_wrapper_t<std::reference_wrapper<double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<double, tt::remove_reference_wrapper_t<double>>,
              "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<const A, tt::remove_reference_wrapper_t<
                                            std::reference_wrapper<const A>>>,
              "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<const A, tt::remove_reference_wrapper_t<const A>>,
    "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<
                  A, tt::remove_reference_wrapper_t<std::reference_wrapper<A>>>,
              "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<A, tt::remove_reference_wrapper_t<A>>,
              "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<double, tt::remove_reference_wrapper_t<
                                 const std::reference_wrapper<double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<double, tt::remove_reference_wrapper_t<
                                 volatile std::reference_wrapper<double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<
                  double, tt::remove_reference_wrapper_t<
                              const volatile std::reference_wrapper<double>>>,
              "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<const double,
                               tt::remove_reference_wrapper_t<
                                   const std::reference_wrapper<const double>>>,
              "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<const double,
                     tt::remove_reference_wrapper_t<
                         volatile std::reference_wrapper<const double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<const double,
                     tt::remove_reference_wrapper_t<
                         const volatile std::reference_wrapper<const double>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<
        A, tt::remove_reference_wrapper_t<const std::reference_wrapper<A>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<
        A, tt::remove_reference_wrapper_t<volatile std::reference_wrapper<A>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<A, tt::remove_reference_wrapper_t<
                            const volatile std::reference_wrapper<A>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<const A, tt::remove_reference_wrapper_t<
                                  const std::reference_wrapper<const A>>>,
    "Failed testing remove_reference_wrapper");
static_assert(
    cpp17::is_same_v<const A, tt::remove_reference_wrapper_t<
                                  volatile std::reference_wrapper<const A>>>,
    "Failed testing remove_reference_wrapper");
static_assert(cpp17::is_same_v<
                  const A, tt::remove_reference_wrapper_t<
                               const volatile std::reference_wrapper<const A>>>,
              "Failed testing remove_reference_wrapper");
/// [remove_reference_wrapper_example]

/// [remove_cvref_wrap]
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<int>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<int&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<const int&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<int&&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<const int&&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<const int>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<volatile int>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<const volatile int>, int>,
    "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<volatile int&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<const volatile int&>, int>,
    "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<tt::remove_cvref_wrap_t<volatile int&&>, int>,
              "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<const volatile int&&>, int>,
    "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<std::reference_wrapper<const int>>,
                     int>,
    "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<std::reference_wrapper<int>>, int>,
    "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<
                  tt::remove_cvref_wrap_t<std::reference_wrapper<int*>>, int*>,
              "Failed testing remove_cvref_wrap");
static_assert(cpp17::is_same_v<
                  tt::remove_cvref_wrap_t<std::reference_wrapper<const int*>>,
                  const int*>,
              "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<
        tt::remove_cvref_wrap_t<std::reference_wrapper<int* const>>, int*>,
    "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<tt::remove_cvref_wrap_t<const std::reference_wrapper<int>>,
                     int>,
    "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<
        tt::remove_cvref_wrap_t<volatile std::reference_wrapper<int>>, int>,
    "Failed testing remove_cvref_wrap");
static_assert(
    cpp17::is_same_v<
        tt::remove_cvref_wrap_t<const volatile std::reference_wrapper<int>>,
        int>,
    "Failed testing remove_cvref_wrap");
/// [remove_cvref_wrap]

