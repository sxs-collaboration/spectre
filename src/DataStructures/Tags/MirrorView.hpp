// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

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

template <typename Tag, typename Space>
struct MirrorView<Tag, Space, DataVector> : db::PrefixTag {
  using tag = Tag;
  using type =
      TensorMetafunctions::swap_type<typename Kokkos::View<double*, Space>,
                                     typename Tag::type>;
};

template <typename Tag, typename Space, typename DataType,
          typename... Properties>
struct MirrorView<Tag, Space, Kokkos::View<DataType, Properties...>>
    : db::PrefixTag {
  using tag = Tag;
  using type = TensorMetafunctions::swap_type<
      typename Kokkos::Impl::MirrorViewType<Space, DataType,
                                            Properties...>::view_type,
      typename Tag::type>;
};

}  // namespace Tags
