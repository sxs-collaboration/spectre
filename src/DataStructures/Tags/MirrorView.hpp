// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS

/// \cond
class DataVector;
/// \endcond

namespace Tags {

/*!
 * \brief Prefix tag representing the given tensor tag in a Kokkos memory space
 *
 * Given a `Tag` that holds a `Tensor`, this tag represents the same tensor in
 * the given Kokkos memory space, or in the default Kokkos memory space if
 * unspecified (see Kokkos documentation on memory spaces). Examples:
 * - Given a `Tag` that holds a `Tensor<DataVector>`, this tag holds a
 *   `Tensor<Kokkos::View<double*>>`, which is allocated in device memory by
 *   default. This is useful to mirror non-Kokkos data to device memory (see
 *   `copy_to_device`).
 * - Given a `Tag` that holds a `Tensor<Kokkos::View<double*>>` and specifying
 *   `Space = Kokkos::HostSpace`, this tag holds a host mirror of the device
 *   data. This is useful for creating a mirror view of the tensor data in host
 *   memory to move data around.
 */
template <typename Tag,
          typename Space = typename Kokkos::DefaultExecutionSpace::memory_space,
          typename VectorType = typename Tag::type::type>
struct MirrorView;

/// \cond
// Specialization for `DataVector` -> mirror to `Kokkos::View<double*>`
// (Omit `Space` in `Kokkos::View<double*>` because that would make it a
// different type. Plain `Kokkos::View<double*>` is what would be hand-written
// or returned by `copy_to_device`. We can add a specialization if custom memory
// spaces are needed.)
template <typename Tag>
struct MirrorView<Tag, typename Kokkos::DefaultExecutionSpace::memory_space,
                  DataVector> : db::PrefixTag,
                                db::SimpleTag {
  using tag = Tag;
  using type =
      TensorMetafunctions::swap_type<Kokkos::View<double*>, typename Tag::type>;
};

template <typename Tag, typename Space, typename DataType,
          typename... Properties>
struct MirrorView<Tag, Space, Kokkos::View<DataType, Properties...>>
    : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = TensorMetafunctions::swap_type<
      // Use public Kokkos API to get the mirror view type
      decltype(Kokkos::create_mirror_view(
          std::declval<Space>(),
          std::declval<Kokkos::View<DataType, Properties...>>())),
      typename Tag::type>;
};
/// \endcond

}  // namespace Tags

#endif  // SPECTRE_KOKKOS
