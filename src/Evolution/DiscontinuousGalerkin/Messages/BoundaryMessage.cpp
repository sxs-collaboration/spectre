// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"

#include <ios>
#include <pup.h>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace evolution::dg {
template <size_t Dim>
BoundaryMessage<Dim>::BoundaryMessage(
    const size_t subcell_ghost_data_size_in, const size_t dg_flux_data_size_in,
    const bool owning_in, const bool enable_if_disabled_in,
    const size_t sender_node_in, const size_t sender_core_in,
    const int tci_status_in, const ::TimeStepId& current_time_step_id_in,
    const ::TimeStepId& next_time_step_id_in,
    const Direction<Dim>& neighbor_direction_in,
    const ElementId<Dim>& element_id_in,
    const Mesh<Dim>& volume_or_ghost_mesh_in,
    const Mesh<Dim - 1>& interface_mesh_in, double* subcell_ghost_data_in,
    double* dg_flux_data_in)
    : subcell_ghost_data_size(subcell_ghost_data_size_in),
      dg_flux_data_size(dg_flux_data_size_in),
      owning(owning_in),
      enable_if_disabled(enable_if_disabled_in),
      sender_node(sender_node_in),
      sender_core(sender_core_in),
      tci_status(tci_status_in),
      current_time_step_id(current_time_step_id_in),
      next_time_step_id(next_time_step_id_in),
      neighbor_direction(neighbor_direction_in),
      element_id(element_id_in),
      volume_or_ghost_mesh(volume_or_ghost_mesh_in),
      interface_mesh(interface_mesh_in),
      subcell_ghost_data(subcell_ghost_data_in),
      dg_flux_data(dg_flux_data_in) {}

template <size_t Dim>
size_t BoundaryMessage<Dim>::total_bytes_with_data(const size_t subcell_size,
                                                   const size_t dg_size) {
  size_t totalsize = sizeof(BoundaryMessage<Dim>);
  totalsize += (subcell_size + dg_size) * sizeof(double);
  return totalsize;
}

template <size_t Dim>
void* BoundaryMessage<Dim>::pack(BoundaryMessage<Dim>* in_msg) {
  // If this is the case, then in_msg is already in the correct memory layout
  // with the data appended to one contiguous buffer (aka owning) so we can just
  // return the message itself
  if (in_msg->owning) {
    return static_cast<void*>(in_msg);
  }

  const size_t subcell_size = in_msg->subcell_ghost_data_size;
  const size_t dg_size = in_msg->dg_flux_data_size;

  const size_t totalsize = total_bytes_with_data(subcell_size, dg_size);

  // The fact that we call the pack() function means we are sending data across
  // address boundaries (nodes) which means we will be owning the data the
  // pointers point to.
  in_msg->owning = true;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* out_msg = reinterpret_cast<BoundaryMessage<Dim>*>(
      CkAllocBuffer(in_msg, static_cast<int>(totalsize)));

  // We cast to char* here to avoid a GCC compiler error
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  memcpy(reinterpret_cast<char*>(out_msg), &in_msg->subcell_ghost_data_size,
         sizeof(BoundaryMessage<Dim>));

  if (subcell_size != 0) {
    // double* + 1 == char* + 8 because double* is 8 bytes
    // Place subcell data right after dg pointer
    out_msg->subcell_ghost_data =
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<double*>(std::addressof(out_msg->dg_flux_data))
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        + 1;
    memcpy(out_msg->subcell_ghost_data, in_msg->subcell_ghost_data,
           subcell_size * sizeof(double));
  }
  if (dg_size != 0) {
    // Place dg data right after subcell data
    out_msg->dg_flux_data =
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<double*>(std::addressof(out_msg->dg_flux_data))
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        + 1 + subcell_size;
    memcpy(out_msg->dg_flux_data, in_msg->dg_flux_data,
           dg_size * sizeof(double));
  }

  // Gotta clean up
  // A possible future optimization is if we know we have to send to another
  // node that we define a BoundaryMessage::allocate function to immediately
  // allocate a message of the right size. This will reduce the number of memory
  // allocations when sending data internode from 3 (2 on send 1 on receive) to
  // 2 (1 on send and 1 on receive).
  delete in_msg;  // NOLINT
  return static_cast<void*>(out_msg);
}

template <size_t Dim>
BoundaryMessage<Dim>* BoundaryMessage<Dim>::unpack(void* in_buf) {
  // We expect in_buf to be in a format where we can interpret it as a
  // BoundaryMessage which is why we can immediately unpack the data as if it
  // was a BoundaryMessage. All we have to do is set the pointers to the
  // beginning of their respective data
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* buffer = reinterpret_cast<BoundaryMessage<Dim>*>(in_buf);

  const size_t subcell_size = buffer->subcell_ghost_data_size;
  const size_t dg_size = buffer->dg_flux_data_size;

  if (subcell_size != 0) {
    // double* + 1 == char* + 8 because double* is 8 bytes
    // Subcell data is located right after dg pointer
    buffer->subcell_ghost_data =
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<double*>(std::addressof(buffer->dg_flux_data))
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        + 1;
  } else {
    buffer->subcell_ghost_data = nullptr;
  }
  if (dg_size != 0) {
    // Dg data is located right after subcell data
    buffer->dg_flux_data =
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<double*>(std::addressof(buffer->dg_flux_data))
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        + 1 + subcell_size;
  } else {
    buffer->dg_flux_data = nullptr;
  }

  // We don't delete in_buf here because it is actually the data we want. We
  // didn't do any new allocations/memcpy's so no need to clean up
  return buffer;
}

template <size_t Dim>
bool operator==(const BoundaryMessage<Dim>& lhs,
                const BoundaryMessage<Dim>& rhs) {
  return lhs.subcell_ghost_data_size == rhs.subcell_ghost_data_size and
         lhs.dg_flux_data_size == rhs.dg_flux_data_size and
         lhs.owning == rhs.owning and
         lhs.enable_if_disabled == rhs.enable_if_disabled and
         lhs.sender_node == rhs.sender_node and
         lhs.sender_core == rhs.sender_core and
         lhs.tci_status == rhs.tci_status and
         lhs.current_time_step_id == rhs.current_time_step_id and
         lhs.next_time_step_id == rhs.next_time_step_id and
         lhs.neighbor_direction == rhs.neighbor_direction and
         lhs.element_id == rhs.element_id and
         lhs.volume_or_ghost_mesh == rhs.volume_or_ghost_mesh and
         lhs.interface_mesh == rhs.interface_mesh and
         // We are guaranteed that lhs.subcell_size == rhs.subcell_size and
         // lhs.dg_size == rhs.dg_size at this point so it's safe to loop over
         // everything
         std::equal(
             lhs.subcell_ghost_data,
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             lhs.subcell_ghost_data + lhs.subcell_ghost_data_size,
             rhs.subcell_ghost_data) and
         std::equal(
             lhs.dg_flux_data,
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             lhs.dg_flux_data + lhs.dg_flux_data_size, rhs.dg_flux_data);
}

template <size_t Dim>
bool operator!=(const BoundaryMessage<Dim>& lhs,
                const BoundaryMessage<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os,
                         const BoundaryMessage<Dim>& message) {
  os << "subcell_ghost_data_size = " << message.subcell_ghost_data_size << "\n";
  os << "dg_flux_data_size = " << message.dg_flux_data_size << "\n";
  os << "owning = " << std::boolalpha << message.owning << "\n";
  os << "enable_if_disabled = " << std::boolalpha << message.enable_if_disabled
     << "\n";
  os << "sender_node = " << message.sender_node << "\n";
  os << "sender_core = " << message.sender_core << "\n";
  os << "tci_status = " << message.tci_status << "\n";
  os << "current_time_ste_id = " << message.current_time_step_id << "\n";
  os << "next_time_ste_id = " << message.next_time_step_id << "\n";
  os << "neighbor_direction = " << message.neighbor_direction << "\n";
  os << "element_id = " << message.element_id << "\n";
  os << "volume_or_ghost_mesh = " << message.volume_or_ghost_mesh << "\n";
  os << "interface_mesh = " << message.interface_mesh << "\n";

  os << "subcell_ghost_data = (";
  if (message.subcell_ghost_data_size > 0) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    os << message.subcell_ghost_data[0];
    for (size_t i = 1; i < message.subcell_ghost_data_size; i++) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      os << "," << message.subcell_ghost_data[i];
    }
  }
  os << ")\n";

  os << "dg_flux_data = (";
  if (message.dg_flux_data_size > 0) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    os << message.dg_flux_data[0];
    for (size_t i = 1; i < message.dg_flux_data_size; i++) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      os << "," << message.dg_flux_data[i];
    }
  }
  os << ")";

  return os;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template struct BoundaryMessage<DIM(data)>;                      \
  template bool operator==(const BoundaryMessage<DIM(data)>& lhs,  \
                           const BoundaryMessage<DIM(data)>& rhs); \
  template bool operator!=(const BoundaryMessage<DIM(data)>& lhs,  \
                           const BoundaryMessage<DIM(data)>& rhs); \
  template std::ostream& operator<<(                               \
      std::ostream& os, const BoundaryMessage<DIM(data)>& message);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg
