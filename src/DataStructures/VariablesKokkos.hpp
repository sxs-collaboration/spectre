// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeSignalingNan.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS

/*!
 * \brief Variables specialization that is backed by a Kokkos::View, meaning
 * it can be allocated on device (GPU) memory.
 *
 * Given a list of tags that hold `Tensor<Kokkos::View<double*>>`, this class
 * allocates a larger `Kokkos::View<double*[num_components]>` to hold all the
 * tensor components in contiguous memory. The memory is either allocated on the
 * host or on a device, depending on the `Kokkos::View` (see Kokkos
 * documentation for details). Individual tensors can be accessed with reference
 * semantics, meaning the `Kokkos::View` returned by `get` are subviews that
 * point into the large memory allocation.
 *
 * This class doesn't support all the arithmetic operations that the `Variables`
 * class backed by a `DataVector` supports. This is because arithmetic
 * operations should be done on device using Kokkos parallel algorithms with
 * precise control over launching kernels on device.
 */
template <typename... Tags>
  requires(sizeof...(Tags) > 0 and
           (Kokkos::is_view<typename Tags::type::type>::value and ...))
class Variables<tmpl::list<Tags...>> {
 public:
  using tags_list = tmpl::list<Tags...>;
  static constexpr auto number_of_variables = sizeof...(Tags);
  static constexpr size_t number_of_independent_components =
      (... + Tags::type::size());

  // The 1D Kokkos::View that backs the tensors, e.g. Kokkos::View<double*>.
  using vector_type = typename tmpl::front<tags_list>::type::type;
  static_assert(vector_type::rank() == 1,
                "Variables can only be used with a 1D Kokkos::View. "
                "A second dimension over tensor components is added "
                "automatically.");
  // The value type of the Kokkos::View, e.g. double.
  using value_type = typename vector_type::value_type;
  // Store the data on device in a 2D Kokkos::View over grid points and tensor
  // components. The two common layouts are:
  // - SoA: tensor components x grid points
  //   (typically better for derivatives because neighboring grid points
  //   are contiguous for each tensor component)
  // - AoS: grid points x tensor components
  //   (typically better for pointwise operations because all tensor components
  //   for a grid point are contiguous)
  // We choose SoA ("LayoutLeft") for now, but this can be changed if needed and
  // should be measured for performance.
  using storage_type =
      Kokkos::View<value_type* [number_of_independent_components],
                   Kokkos::LayoutLeft, typename vector_type::memory_space>;
  // The Kokkos memory space the data lives in, e.g. Kokkos::CudaSpace.
  using memory_space = typename vector_type::memory_space;

  Variables() = default;
  explicit Variables(const size_t number_of_grid_points) {
    initialize(number_of_grid_points);
  }
  explicit Variables(storage_type&& rhs) : storage_(std::move(rhs)) {
    set_reference_variable_data();
  }
  ~Variables() = default;

  /// Initialize the memory for the Variables. You can specify a `memory_space`
  /// to allocate the memory on a specific GPU. Does nothing if the number of
  /// grid points is unchanged. Fills the memory with NaN only if
  /// SPECTRE_NAN_INIT is defined.
  void initialize(const size_t number_of_grid_points,
                  const memory_space& space = {}) {
    if (number_of_grid_points == this->number_of_grid_points()) {
      return;
    }
    storage_ = storage_type(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Variables", space),
        number_of_grid_points);
#if defined(SPECTRE_NAN_INIT)
    Kokkos::deep_copy(storage_, make_signaling_NaN<value_type>());
#endif  // SPECTRE_NAN_INIT
    set_reference_variable_data();
  }

  constexpr SPECTRE_ALWAYS_INLINE size_t size() const {
    return storage_.size();
  }
  constexpr SPECTRE_ALWAYS_INLINE size_t number_of_grid_points() const {
    return storage_.extent(0);
  }

  storage_type& view() { return storage_; }
  const storage_type& view() const { return storage_; }

  template <typename Tag, typename TagList>
  friend constexpr typename Tag::type& get(  // NOLINT
      Variables<TagList>& v);
  template <typename Tag, typename TagList>
  friend constexpr const typename Tag::type& get(  // NOLINT
      const Variables<TagList>& v);

  template <typename Space>
  auto create_mirror_view(const Space& space) const {
    return Variables<tmpl::list<::Tags::MirrorView<Tags, Space>...>>(
        Kokkos::create_mirror_view(space, storage_));
  }

  template <typename Space>
  auto create_mirror_view_and_copy(const Space& space) const {
    return Variables<tmpl::list<::Tags::MirrorView<Tags, Space>...>>(
        Kokkos::create_mirror_view_and_copy(space, storage_));
  }

 private:
  void set_reference_variable_data() {
    size_t variable_offset = 0;
    tmpl::for_each<tags_list>([this, &variable_offset](auto tag_v) {
      using Tag = tmpl::type_from<decltype(tag_v)>;
      auto& var = tuples::get<Tag>(reference_variable_data_);
      for (size_t i = 0; i < Tag::type::size(); ++i) {
        // The subview is contiguous if each tensor component is contiguous in
        // memory, which is the case for Kokkos::LayoutLeft (SoA). Otherwise,
        // the subview has a stride.
        var[i] = Kokkos::subview(storage_, Kokkos::ALL(), variable_offset);
        ++variable_offset;
      }
    });
  }

  storage_type storage_{};
  tuples::TaggedTuple<Tags...> reference_variable_data_;
};

/// Get each tensor at the given grid-point index.
template <typename... VarTags, typename... Is>
KOKKOS_FUNCTION auto make_at_index(
    const Variables<tmpl::list<VarTags...>>& vars, const Is&... i) {
  tuples::TaggedTuple<::Tags::AtIndex<VarTags>...> result{};
  (..., (get<::Tags::AtIndex<VarTags>>(result) =
             make_at_index(get<VarTags>(vars), i...)));
  return result;
}

/// @{
/// Set each tensor at the given grid-point index.
template <typename... VarTags, typename... Is>
KOKKOS_FUNCTION void set_at_index(
    const gsl::not_null<Variables<tmpl::list<VarTags...>>*> vars,
    const tuples::TaggedTuple<::Tags::AtIndex<VarTags>...>& value,
    const Is&... i) {
  (..., (set_at_index(make_not_null(&get<VarTags>(*vars)),
                      get<::Tags::AtIndex<VarTags>>(value), i...)));
}
template <typename... VarTags, typename... Is>
KOKKOS_FUNCTION void set_at_index(
    const gsl::not_null<const Variables<tmpl::list<VarTags...>>*> vars,
    const tuples::TaggedTuple<::Tags::AtIndex<VarTags>...>& value,
    const Is&... i) {
  (..., (set_at_index(make_not_null(&get<VarTags>(*vars)),
                      get<::Tags::AtIndex<VarTags>>(value), i...)));
}
/// @}

/// Allocate a `Variables` of the same size as `vars_host` in device memory,
/// without copying the data yet. You can specify a `memory_space` to allocate
/// the memory on a specific GPU.
template <typename... Tags>
auto allocate_on_device(
    const Variables<tmpl::list<Tags...>>& vars_host,
    const typename Kokkos::DefaultExecutionSpace::memory_space& space = {}) {
  static_assert((std::is_same_v<typename Tags::type::type, DataVector> and ...),
                "Only a DataVector-backed Variables can be mirrored in device "
                "memory.");
  Variables<tmpl::list<::Tags::MirrorView<Tags>...>> vars_device{};
  if constexpr (sizeof...(Tags) > 0) {
    vars_device.initialize(vars_host.number_of_grid_points(), space);
  }
  return vars_device;
}

/// Copy the `vars_host` to an existing `Variables` in device memory. The device
/// memory is only reallocated if the number of grid points changed.
template <typename DeviceTags, typename... Tags>
void copy_to_device(const gsl::not_null<Variables<DeviceTags>*> vars_device,
                    const Variables<tmpl::list<Tags...>>& vars_host) {
  static_assert((std::is_same_v<typename Tags::type::type, DataVector> and ...),
                "Only a DataVector-backed Variables can be copied to device "
                "memory.");
  if constexpr (sizeof...(Tags) > 0) {
    using HostVars = Variables<tmpl::list<Tags...>>;
    using ValueType = typename HostVars::value_type;
    static constexpr size_t num_components =
        HostVars::number_of_independent_components;
    const size_t num_points = vars_host.number_of_grid_points();
    if (vars_device->number_of_grid_points() != num_points) {
      vars_device->initialize(num_points);
    }
    // View that wraps the host data (DataVector-backed Variables) in a
    // Kokkos::View without owning it so we can deep-copy the data. Relies on
    // the memory layout in the DataVector-backed Variables being the same as
    // the Kokkos view (LayoutLeft).
    using HostUnmanaged =
        Kokkos::View<const ValueType* [num_components], Kokkos::LayoutLeft,
                     Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
    HostUnmanaged unmanaged_host_view(vars_host.data(), num_points);
    // Copy to device
    Kokkos::deep_copy(vars_device->view(), unmanaged_host_view);
  }
}

/// Copy the `vars_host` to device memory. You can specify a `memory_space` to
/// allocate the memory on a specific GPU.
template <typename... Tags>
auto copy_to_device(
    const Variables<tmpl::list<Tags...>>& vars_host,
    const typename Kokkos::DefaultExecutionSpace::memory_space& space = {}) {
  auto vars_device = allocate_on_device(vars_host, space);
  copy_to_device(make_not_null(&vars_device), vars_host);
  return vars_device;
}

/// Copy a `Variables` from device to host memory.
template <typename HostTags, typename DeviceTags>
void copy_to_host(const gsl::not_null<Variables<HostTags>*> vars_host,
                  const Variables<DeviceTags>& vars_device) {
  if constexpr (tmpl::size<DeviceTags>::value > 0) {
    using HostVars = Variables<HostTags>;
    static_assert(std::is_same_v<typename HostVars::vector_type, DataVector>,
                  "Only a DataVector-backed Variables can be copied from "
                  "device memory.");
    using ValueType = typename HostVars::value_type;
    static constexpr size_t num_components =
        HostVars::number_of_independent_components;
    const size_t num_points = vars_device.number_of_grid_points();
    // View that wraps the host data (DataVector-backed Variables) in a
    // Kokkos::View without owning it so we can deep-copy the data. Relies on
    // the memory layout in the DataVector-backed Variables being the same as
    // the Kokkos view (LayoutLeft).
    using HostUnmanaged =
        Kokkos::View<ValueType* [num_components], Kokkos::LayoutLeft,
                     Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
    if (vars_host->number_of_grid_points() != num_points) {
      vars_host->initialize(num_points);
    }
    HostUnmanaged unmanaged_host_view(vars_host->data(), num_points);
    // Copy to host
    Kokkos::deep_copy(unmanaged_host_view, vars_device.view());
  }
}

#endif  // SPECTRE_KOKKOS
