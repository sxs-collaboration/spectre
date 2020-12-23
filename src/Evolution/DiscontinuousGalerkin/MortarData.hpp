// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <ostream>
#include <pup.h>
#include <utility>
#include <vector>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace evolution::dg {
/*!
 * \brief Data on the mortar used to compute the boundary correction for the
 * DG scheme.
 *
 * The class holds the local data that has been projected to the mortar as well
 * as the neighbor data that has been projected to the mortar. The local and
 * neighbor data is later used to compute the same unique boundary correction on
 * the mortar for both elements. That is, the final boundary correction
 * computation is done twice: once on each element touching the mortar. However,
 * the computation is done in such a way that the results agree.
 *
 * In addition to the (type-erased) fields on both sides of the mortar, the face
 * (not mortar!) mesh of the neighbor is stored. The mesh will be necessary
 * when hybridizing DG with finite difference or finite volume schemes
 * (DG-subcell).
 *
 * If the element and its neighbor have unaligned logical coordinate systems
 * then the data is stored in the local logical coordinate's orientation
 * (\f$\xi\f$ varies fastest). This means the action sending the data is
 * responsible for reorienting the data on the mortar so it matches the
 * neighbor's orientation.
 *
 * \tparam Dim the volume dimension of the mesh
 */
template <size_t Dim>
class MortarData {
 public:
  /*!
   * \brief Insert data onto the mortar.
   *
   * Exactly one local and neighbor insert call must be made between calls to
   * `extract()`.
   *
   * The insert functions require that:
   * - the data is inserted only once
   * - the `TimeStepId` of the local and neighbor data are the same (this is
   *   only checked if the local/neighbor data was already inserted)
   *
   * \note it is not required that the number of grid points between the local
   * and neighbor data be the same since one may be using FD/FV instead of DG
   * and this switch is done locally in space and time in such a way that
   * neighboring elements have no a priori knowledge about what well be
   * received.
   */
  //@{
  void insert_local_mortar_data(TimeStepId time_step_id,
                                Mesh<Dim - 1> local_interface_mesh,
                                std::vector<double> local_mortar_vars) noexcept;
  void insert_neighbor_mortar_data(
      TimeStepId time_step_id, Mesh<Dim - 1> neighbor_interface_mesh,
      std::vector<double> neighbor_mortar_vars) noexcept;
  //@}

  /// Return the inserted data and reset the state to empty.
  ///
  /// The first element is the local data while the second element is the
  /// neighbor data.
  auto extract() noexcept
      -> std::pair<std::pair<Mesh<Dim - 1>, std::vector<double>>,
                   std::pair<Mesh<Dim - 1>, std::vector<double>>>;

  const TimeStepId& time_step_id() const noexcept { return time_step_id_; }

  auto local_mortar_data() const noexcept
      -> const std::optional<std::pair<Mesh<Dim - 1>, std::vector<double>>>& {
    return local_mortar_data_;
  }

  auto neighbor_mortar_data() const noexcept
      -> const std::optional<std::pair<Mesh<Dim - 1>, std::vector<double>>>& {
    return neighbor_mortar_data_;
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  TimeStepId time_step_id_{};
  std::optional<std::pair<Mesh<Dim - 1>, std::vector<double>>>
      local_mortar_data_{};
  std::optional<std::pair<Mesh<Dim - 1>, std::vector<double>>>
      neighbor_mortar_data_{};
};

template <size_t Dim>
bool operator==(const MortarData<Dim>& lhs,
                const MortarData<Dim>& rhs) noexcept;
template <size_t Dim>
bool operator!=(const MortarData<Dim>& lhs,
                const MortarData<Dim>& rhs) noexcept;
}  // namespace evolution::dg
