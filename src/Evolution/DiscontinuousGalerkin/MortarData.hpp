// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <pup.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

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
  using MortarType = std::optional<std::pair<Mesh<Dim - 1>, DataVector>>;

 public:
  MortarData(size_t number_of_buffers = 1);

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
  /// @{
  void insert_local_mortar_data(TimeStepId time_step_id,
                                Mesh<Dim - 1> local_interface_mesh,
                                DataVector local_mortar_vars);
  void insert_neighbor_mortar_data(TimeStepId time_step_id,
                                   Mesh<Dim - 1> neighbor_interface_mesh,
                                   DataVector neighbor_mortar_vars);
  /// @}

  /*!
   * \brief Insert the magnitude of the local face normal, the determinant
   * of the volume inverse Jacobian, and the determinant of the face Jacobian.
   * Used for local time stepping with Gauss points.
   *
   * The magnitude of the face normal is given by:
   *
   * \f{align*}{
   *  \sqrt{
   *   \frac{\partial\xi}{\partial x^i} \gamma^{ij}
   *   \frac{\partial\xi}{\partial x^j}}
   * \f}
   *
   * for a face in the \f$\xi\f$-direction, with inverse spatial metric
   * \f$\gamma^{ij}\f$.
   */
  void insert_local_geometric_quantities(
      const Scalar<DataVector>& local_volume_det_inv_jacobian,
      const Scalar<DataVector>& local_face_det_jacobian,
      const Scalar<DataVector>& local_face_normal_magnitude);

  /*!
   * \brief Insert the magnitude of the local face normal. Used for local time
   * stepping with Gauss-Lobatto points.
   *
   * The magnitude of the face normal is given by:
   *
   * \f{align*}{
   *  \sqrt{
   *   \frac{\partial\xi}{\partial x^i} \gamma^{ij}
   *   \frac{\partial\xi}{\partial x^j}}
   * \f}
   *
   * for a face in the \f$\xi\f$-direction, with inverse spatial metric
   * \f$\gamma^{ij}\f$.
   */
  void insert_local_face_normal_magnitude(
      const Scalar<DataVector>& local_face_normal_magnitude);

  /*!
   * \brief Sets the `local_volume_det_inv_jacobian` by setting the DataVector
   * to point into the `MortarData`'s internal storage.
   *
   * \warning The result should never be changed.
   */
  void get_local_volume_det_inv_jacobian(
      gsl::not_null<Scalar<DataVector>*> local_volume_det_inv_jacobian) const;

  /*!
   * \brief Sets the `local_face_det_jacobian` by setting the DataVector to
   * point into the `MortarData`'s internal storage.
   *
   * \warning The result should never be changed.
   */
  void get_local_face_det_jacobian(
      gsl::not_null<Scalar<DataVector>*> local_face_det_jacobian) const;

  /*!
   * \brief Sets the `local_face_normal_magnitude` by setting the DataVector to
   * point into the `MortarData`'s internal storage.
   *
   * \warning The result should never be changed.
   */
  void get_local_face_normal_magnitude(
      gsl::not_null<Scalar<DataVector>*> local_face_normal_magnitude) const;

  /// Return the inserted data and reset the state to empty.
  ///
  /// The first element is the local data while the second element is the
  /// neighbor data.
  auto extract() -> std::pair<std::pair<Mesh<Dim - 1>, DataVector>,
                              std::pair<Mesh<Dim - 1>, DataVector>>;

  /// Move to the next internal mortar buffer
  void next_buffer();

  /// Return the current internal mortar index
  size_t current_buffer_index() const;

  /// Return the total number of buffers that this MortarData was constructed
  /// with
  size_t total_number_of_buffers() const;

  const TimeStepId& time_step_id() const {
    return time_step_id_[mortar_index_];
  }

  TimeStepId& time_step_id() { return time_step_id_[mortar_index_]; }

  auto local_mortar_data() const
      -> const std::optional<std::pair<Mesh<Dim - 1>, DataVector>>& {
    return local_mortar_data_[mortar_index_];
  }

  auto neighbor_mortar_data() const
      -> const std::optional<std::pair<Mesh<Dim - 1>, DataVector>>& {
    return neighbor_mortar_data_[mortar_index_];
  }

  auto local_mortar_data()
      -> std::optional<std::pair<Mesh<Dim - 1>, DataVector>>& {
    return local_mortar_data_[mortar_index_];
  }

  auto neighbor_mortar_data()
      -> std::optional<std::pair<Mesh<Dim - 1>, DataVector>>& {
    return neighbor_mortar_data_[mortar_index_];
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE
  friend bool operator==(const MortarData<LocalDim>& lhs,
                         const MortarData<LocalDim>& rhs);

  size_t number_of_buffers_{1};
  std::vector<TimeStepId> time_step_id_{};
  std::vector<MortarType> local_mortar_data_{};
  std::vector<MortarType> neighbor_mortar_data_{};
  size_t mortar_index_{0};
  DataVector local_geometric_quantities_{};
  bool using_volume_and_face_jacobians_{false};
  bool using_only_face_normal_magnitude_{false};
};

template <size_t Dim>
bool operator!=(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs);

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const MortarData<Dim>& mortar_data);
}  // namespace evolution::dg
