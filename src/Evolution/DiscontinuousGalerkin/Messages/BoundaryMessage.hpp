// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>
#include <type_traits>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GetOutput.hpp"
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

  // Needed for charm registration
  static std::string name() {
    return "BoundaryMessage<" + get_output(Dim) + ">";
  };

  size_t subcell_ghost_data_size;
  size_t dg_flux_data_size;
  // Whether or not this BoundaryMessage owns the data that the subcell and dg
  // pointers point to
  bool owning;
  bool enable_if_disabled;
  size_t sender_node;
  size_t sender_core;
  int tci_status;
  ::TimeStepId current_time_step_id;
  ::TimeStepId next_time_step_id;
  Direction<Dim> neighbor_direction;
  ElementId<Dim> element_id;
  Mesh<Dim> volume_or_ghost_mesh;
  Mesh<Dim - 1> interface_mesh;

  // If set to nullptr then we aren't sending that type of data.
  double* subcell_ghost_data;
  double* dg_flux_data;

  BoundaryMessage() = default;

  BoundaryMessage(const size_t subcell_ghost_data_size_in,
                  const size_t dg_flux_data_size_in, const bool owning_in,
                  const bool enable_if_disabled_in, const size_t sender_node_in,
                  const size_t sender_core_in, const int tci_status_in,
                  const ::TimeStepId& current_time_step_id_in,
                  const ::TimeStepId& next_time_step_id_in,
                  const Direction<Dim>& neighbor_direction_in,
                  const ElementId<Dim>& element_id_in,
                  const Mesh<Dim>& volume_or_ghost_mesh_in,
                  const Mesh<Dim - 1>& interface_mesh_in,
                  double* subcell_ghost_data_in, double* dg_flux_data_in);

  /*!
   * \brief This is the size (in bytes) necessary to allocate a BoundaryMessage
   * including the arrays of data as well.
   *
   * This will add `(subcell_size + dg_size) * sizeof(double)` number of bytes
   * to `sizeof(BoundaryMessage<Dim>)`.
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

}  // namespace evolution::dg

#define CK_TEMPLATES_ONLY
#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.def.h"
#undef CK_TEMPLATES_ONLY
