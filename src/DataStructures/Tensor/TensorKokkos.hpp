// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// Copy a `Tensor<DataVector>` to device memory.
template <typename HostVectorType, typename... Properties>
auto copy_to_device(const Tensor<HostVectorType, Properties...>& tensor_host) {
  using ValueType = typename HostVectorType::value_type;
  using DeviceVectorType = Kokkos::View<ValueType*>;
  Tensor<DeviceVectorType, Properties...> tensor_device{
      "Tensor", tensor_host.begin()->size()};
  // View that wraps the host data (DataVector) in a Kokkos::View without owning
  // it so we can deep-copy the data
  using HostUnmanaged = Kokkos::View<const ValueType*, Kokkos::HostSpace,
                                     Kokkos::MemoryUnmanaged>;
  for (size_t i = 0; i < tensor_host.size(); ++i) {
    HostUnmanaged unmanaged_host_view(tensor_host[i].data(),
                                      tensor_host[i].size());
    // Copy to device
    Kokkos::deep_copy(tensor_device[i], unmanaged_host_view);
  }
  return tensor_device;
}

/// Copy a `Tensor<Kokkos::View>` to host memory.
template <typename DeviceVectorType, typename... Properties>
auto copy_to_host(
    const Tensor<DeviceVectorType, Properties...>& tensor_device) {
  using ValueType = typename DeviceVectorType::value_type;
  static_assert(
      std::is_same_v<ValueType, double>,
      "Only Tensor<Kokkos::View<double*>> is currently supported. Other "
      "datatypes can be added by mapping to their host representation (e.g. "
      "complex -> ComplexDataVector).");
  using HostVectorType = DataVector;
  Tensor<HostVectorType, Properties...> tensor_host{
      tensor_device.begin()->size()};
  // View that wraps the host data (DataVector) in a Kokkos::View without owning
  // it so we can deep-copy the data
  using HostUnmanaged =
      Kokkos::View<ValueType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  for (size_t i = 0; i < tensor_device.size(); ++i) {
    HostUnmanaged unmanaged_host_view(tensor_host[i].data(),
                                      tensor_host[i].size());
    // Copy to host
    Kokkos::deep_copy(unmanaged_host_view, tensor_device[i]);
  }
  return tensor_host;
}
#endif  // SPECTRE_KOKKOS
