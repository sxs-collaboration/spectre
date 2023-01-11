// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>
#include <type_traits>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/PrettyType.hpp"

#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.decl.h"

namespace evolution::dg {
/*!
 * \brief [Charm++ Message]
 * (https://charm.readthedocs.io/en/latest/charm%2B%2B/manual.html#messages)
 * intended to be used in `receive_data` calls on the elements to send boundary
 * data from one element on one node, to a different element on a (potentially)
 * different node.
 *
 * If this message is to be sent across nodes, the `pack()` and `unpack()`
 * methods will be called on the sending and receiving node, respectively.
 */
template <size_t Dim>
struct BoundaryMessage : public CMessage_BoundaryMessage<Dim> {
  using base = CMessage_BoundaryMessage<Dim>;

  size_t subcell_ghost_data_size;
  size_t dg_flux_data_size;
  bool sent_across_nodes;
  size_t sender_node;
  size_t sender_core;
  ::TimeStepId current_time_step_id;
  ::TimeStepId next_time_step_id;
  Mesh<Dim> volume_or_ghost_mesh;
  Mesh<Dim - 1> interface_mesh;

  // If set to nullptr then we aren't sending that type of data.
  double* subcell_ghost_data;
  double* dg_flux_data;

  BoundaryMessage() = default;

  BoundaryMessage(const size_t subcell_ghost_data_size_in,
                  const size_t dg_flux_data_size_in,
                  const bool sent_across_nodes_in, const size_t sender_node_in,
                  const size_t sender_core_in,
                  const ::TimeStepId& current_time_step_id_in,
                  const ::TimeStepId& next_time_step_id_in,
                  const Mesh<Dim>& volume_or_ghost_mesh_in,
                  const Mesh<Dim - 1>& interface_mesh_in,
                  double* subcell_ghost_data_in, double* dg_flux_data_in);

  /*!
   * \brief This is the size (in bytes) necessary to allocate a BoundaryMessage
   * with only the member variables (no actual subcell/dg data, only pointers).
   *
   * This evaluates to 256 bytes for Dim == 1, 280 bytes for Dim == 2, and 312
   * bytes for Dim == 3.
   */
  static size_t total_bytes_without_data();
  /*!
   * \brief This is the size (in bytes) necessary to allocate a BoundaryMessage
   * including the arrays of data as well.
   *
   * This will add `(subcell_size + dg_size) * sizeof(double)` number of bytes
   * to `total_bytes_without_data()`.
   */
  static size_t total_bytes_with_data(const size_t subcell_size,
                                      const size_t dg_size);

  static void* pack(BoundaryMessage*);
  static BoundaryMessage* unpack(void*);
};

template <size_t Dim>
bool operator==(const BoundaryMessage<Dim>& lhs,
                const BoundaryMessage<Dim>& rhs);
template <size_t Dim>
bool operator!=(const BoundaryMessage<Dim>& lhs,
                const BoundaryMessage<Dim>& rhs);

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const BoundaryMessage<Dim>& message);

namespace detail {
template <typename T>
size_t offset() {
  if constexpr (std::is_same_v<T, size_t>) {
    return sizeof(size_t);
  } else if constexpr (std::is_same_v<T, bool>) {
    // BoundaryMessage is 8-byte aligned so a bool isn't 1 byte, it's actually 8
    return 8;
  } else if constexpr (std::is_same_v<T, TimeStepId>) {
    return sizeof(TimeStepId);
  } else if constexpr (std::is_same_v<T, Mesh<3>> or
                       std::is_same_v<T, Mesh<2>> or
                       std::is_same_v<T, Mesh<1>>) {
    return sizeof(T);
  } else if constexpr (std::is_same_v<T, Mesh<0>>) {
    // Mesh<0> is only 3 bytes, but we need 8 for alignment
    return 8;
  } else if constexpr (std::is_same_v<T, double*>) {
    return sizeof(double*);
  } else {
    ERROR("Cannot calculate offset for '"
          << pretty_type::name<T>()
          << "' in a BoundaryMessage. Offset is only known for size_t, bool, "
             "TimeStepId, Mesh, double*.");
    return 0;
  }
}
}  // namespace detail
}  // namespace evolution::dg

#define CK_TEMPLATES_ONLY
#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.def.h"
#undef CK_TEMPLATES_ONLY
