// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "Utilities/ErrorHandling/DeviceMessage.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

/// \cond
// Forward declarations of the types the trait is specialized for, so it doesn't
// include their headers (and `Utilities/Gsl.hpp` can use it)
namespace gsl {
template <class T>
class not_null;
template <class ElementType, std::ptrdiff_t Extent>
class span;
}  // namespace gsl
template <typename X, typename Symm, typename IndexList>
class Tensor;
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
template <typename TagList, bool is_fundamental>
struct TempBuffer;
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace Kokkos {
template <class DataType, class... Properties>
class View;
}  // namespace Kokkos
#endif  // SPECTRE_KOKKOS

namespace tt {
/// @{
/*!
 * \ingroup TypeTraitsGroup
 * \brief Whether objects of type `T` can be used in device code, e.g. as
 * arguments of a `KOKKOS_FUNCTION` that runs on a GPU.
 *
 * By default these are the trivially copyable, non-polymorphic types, i.e. the
 * types that can be copied into a Kokkos kernel byte by byte and have no
 * virtual functions. This includes `double`, `size_t`, `Tensor`s of `double`s,
 * `Mesh`, `Kokkos:View`s, and lambdas. It excludes types that own host
 * memory, such as `DataVector`, `Tensor<DataVector>`, `std::vector` and
 * `std::string`, and polymorphic classes, such as the host equations of
 * state. Pointers, `gsl::not_null`, `gsl::span`, arrays, `std::array`,
 * `std::pair`, `std::tuple`, `std::optional`, `Tensor`s, `tuples::TaggedTuple`s
 * and `TempBuffer`s are device-capable if the types they point to or hold are.
 * `std::complex` is not, because it gives wrong results on  GPUs (see \ref
 * gpu_support).
 *
 * Specialize `tt::is_device_capable` to change the default for a type, e.g.
 * for a class that holds a `Kokkos::View` and is copied into kernels.
 *
 * This only checks the type, not where the data is: a `double*` pointing to
 * host memory is device-capable by this definition. It also can't check
 * whether a callable such as a lambda can run on the device.
 *
 * \see ASSERT_DEVICE_CAPABLE
 */
template <typename T>
struct is_device_capable
    : std::bool_constant<std::is_trivially_copyable_v<T> and
                         not std::is_polymorphic_v<T>> {};

template <typename... Ts>
constexpr bool is_device_capable_v =
    (is_device_capable<std::remove_cvref_t<Ts>>::value and ...);
/// @}

/// \cond
template <typename T>
struct is_device_capable<T*>
    : std::bool_constant<std::is_void_v<T> or is_device_capable_v<T>> {};

template <typename T, size_t N>
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
struct is_device_capable<T[N]> : std::bool_constant<is_device_capable_v<T>> {};

template <typename T>
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
struct is_device_capable<T[]> : std::bool_constant<is_device_capable_v<T>> {};

template <typename T, size_t N>
struct is_device_capable<std::array<T, N>>
    : std::bool_constant<is_device_capable_v<T>> {};

// `std::pair` and `std::tuple` aren't trivially copyable even if their elements
// are: they define their own assignment operators (to assign through reference
// members, e.g. with `std::tie`, and from pairs and tuples of other types), and
// libstdc++ also defines the move constructor of `std::tuple`. Their copy
// constructors are trivial though, so they can be copied into kernels, and
// their (constexpr) member functions gave correct results on GPUs. So they are
// device-capable if their elements are.
template <typename T, typename U>
struct is_device_capable<std::pair<T, U>>
    : std::bool_constant<is_device_capable_v<T, U>> {};

template <typename... Ts>
struct is_device_capable<std::tuple<Ts...>>
    : std::bool_constant<is_device_capable_v<Ts...>> {};

template <typename T>
struct is_device_capable<std::optional<T>>
    : std::bool_constant<is_device_capable_v<T>> {};

template <typename T>
struct is_device_capable<std::complex<T>> : std::false_type {};

template <typename T>
struct is_device_capable<gsl::not_null<T>>
    : std::bool_constant<is_device_capable_v<T>> {};

template <typename T, std::ptrdiff_t Extent>
struct is_device_capable<gsl::span<T, Extent>>
    : std::bool_constant<is_device_capable_v<T>> {};

// E.g. `Tensor`s of `Kokkos::View`s, which aren't trivially copyable
template <typename X, typename Symm, typename IndexList>
struct is_device_capable<Tensor<X, Symm, IndexList>>
    : std::bool_constant<is_device_capable_v<X>> {};

// `TaggedTuple`s aren't trivially copyable for the same reason as
// `std::tuple`s (see above), but their copy constructors are trivial if those
// of their elements are
template <typename... Tags>
struct is_device_capable<tuples::TaggedTuple<Tags...>>
    : std::bool_constant<is_device_capable_v<typename Tags::type...>> {};

// The `TempBuffer` for fundamental types is a `TaggedTuple`. (The one for
// `DataVector`s is a `Variables`, which isn't device-capable.)
template <template <typename...> class List, typename... Tags>
struct is_device_capable<TempBuffer<List<Tags...>, true>>
    : std::bool_constant<is_device_capable_v<typename Tags::type...>> {};

#ifdef SPECTRE_KOKKOS
template <typename DataType, typename... Properties>
struct is_device_capable<Kokkos::View<DataType, Properties...>>
    : std::true_type {};
#endif  // SPECTRE_KOKKOS
/// \endcond
}  // namespace tt

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Aborts if the function is running on the device and any of the types
 * passed to the macro aren't `tt::is_device_capable`.
 *
 * Every function marked `SPECTRE_KOKKOS_DISABLE_EXEC_CHECK` must call this at
 * the start of its body with the template parameters (or other types) that
 * can be host-only, e.g.
 *
 * \code
 * SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
 * template <typename DataType>
 * KOKKOS_FUNCTION Scalar<DataType> f(const Scalar<DataType>& x) {
 *   ASSERT_DEVICE_CAPABLE(DataType);
 *   ...
 * }
 * \endcode
 *
 * Disabling the exec check stops nvcc from diagnosing calls of host-only
 * instantiations, such as `f<DataVector>`, from device code, so this check
 * catches them at runtime instead. It can't be a `static_assert`, because nvcc
 * compiles the host-only instantiations for the device as well, even if they
 * are only called on the host. For device-capable types the check is removed
 * at compile time, and it is never done on the host, so it costs nothing. It is
 * active in all build types, unlike `ASSERT`.
 *
 * Callables can't be checked (see `tt::is_device_capable`), so don't pass the
 * types of functions or lambdas, only the types of data.
 */
#ifdef SPECTRE_KOKKOS
#define ASSERT_DEVICE_CAPABLE(...)                                          \
  do {                                                                      \
    KOKKOS_IF_ON_DEVICE((if constexpr (not ::tt::is_device_capable_v<       \
                                           __VA_ARGS__>) {                  \
      SPECTRE_ABORT_ON_DEVICE(                                              \
          "ASSERT FAILED", "tt::is_device_capable_v<" #__VA_ARGS__ ">",     \
          "This function was called on the device with a type that can "    \
          "only be used on the host. The template arguments are listed in " \
          "the function signature above.");                                 \
    }))                                                                     \
  } while (false)
#else
#define ASSERT_DEVICE_CAPABLE(...) \
  do {                             \
  } while (false)
#endif  // SPECTRE_KOKKOS
