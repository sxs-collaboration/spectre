// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"

#include <cstddef>
#include <optional>
#include <ostream>
#include <pup.h>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace evolution::dg {
template <size_t Dim>
MortarData<Dim>::MortarData(const size_t number_of_buffers)
    : number_of_buffers_(number_of_buffers) {
  time_step_id_.resize(number_of_buffers_);
  local_mortar_data_.resize(number_of_buffers_);
  neighbor_mortar_data_.resize(number_of_buffers_);
  mortar_index_ = 0;
}

template <size_t Dim>
void MortarData<Dim>::insert_local_mortar_data(
    TimeStepId time_step_id, Mesh<Dim - 1> local_interface_mesh,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    DataVector local_mortar_vars) {
  // clang-tidy can't figure out that `vars` is moved below
  ASSERT(not local_mortar_data_[mortar_index_].has_value(),
         "Already received local data at " << time_step_id
                                           << " with interface mesh "
                                           << local_interface_mesh);
  ASSERT(not neighbor_mortar_data_[mortar_index_].has_value() or
             time_step_id == time_step_id_[mortar_index_],
         "Received local data at " << time_step_id
                                   << ", but already have neighbor data at "
                                   << time_step_id_[mortar_index_]);
  // NOLINTNEXTLINE(performance-move-const-arg)
  time_step_id_[mortar_index_] = std::move(time_step_id);
  local_mortar_data_[mortar_index_] =
      std::pair{std::move(local_interface_mesh), std::move(local_mortar_vars)};
}

template <size_t Dim>
void MortarData<Dim>::insert_neighbor_mortar_data(
    TimeStepId time_step_id, Mesh<Dim - 1> neighbor_interface_mesh,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    DataVector neighbor_mortar_vars) {
  // clang-tidy can't figure out that `vars` is moved below
  ASSERT(not neighbor_mortar_data_[mortar_index_].has_value(),
         "Already received neighbor data at " << time_step_id
                                              << " with interface mesh "
                                              << neighbor_interface_mesh);
  ASSERT(not local_mortar_data_[mortar_index_].has_value() or
             time_step_id == time_step_id_[mortar_index_],
         "Received neighbor data at " << time_step_id
                                      << ", but already have local data at "
                                      << time_step_id_[mortar_index_]);
  // NOLINTNEXTLINE(performance-move-const-arg)
  time_step_id_[mortar_index_] = std::move(time_step_id);
  neighbor_mortar_data_[mortar_index_] = std::pair{
      std::move(neighbor_interface_mesh), std::move(neighbor_mortar_vars)};
}

template <size_t Dim>
void MortarData<Dim>::insert_local_geometric_quantities(
    const Scalar<DataVector>& local_volume_det_inv_jacobian,
    const Scalar<DataVector>& local_face_det_jacobian,
    const Scalar<DataVector>& local_face_normal_magnitude) {
  ASSERT(local_mortar_data_[mortar_index_].has_value(),
         "Must set local mortar data before setting the geometric quantities.");
  ASSERT(local_face_det_jacobian[0].size() ==
             local_face_normal_magnitude[0].size(),
         "The determinant of the local face Jacobian has "
             << local_face_det_jacobian[0].size()
             << " grid points, and the magnitude of the local face normal has "
             << local_face_normal_magnitude[0].size()
             << " but they must be the same");
  ASSERT(local_face_det_jacobian[0].size() ==
             std::get<0>(*local_mortar_data()).number_of_grid_points(),
         "The number of grid points ("
             << std::get<0>(*local_mortar_data()).number_of_grid_points()
             << ") on the local face must match the number of grid points "
                "passed in for the face Jacobian determinant and normal vector "
                "magnitude ("
             << local_face_det_jacobian[0].size() << ")");
  ASSERT(not using_only_face_normal_magnitude_,
         "The face normal, volume inverse Jacobian determinant, and face "
         "Jacobian determinant cannot be inserted because the only the face "
         "normal is being used.");
  using_volume_and_face_jacobians_ = true;
  const size_t required_storage_size = local_volume_det_inv_jacobian[0].size() +
                                       2 * local_face_det_jacobian[0].size();
  local_geometric_quantities_.destructive_resize(required_storage_size);

  std::copy(local_volume_det_inv_jacobian[0].begin(),
            local_volume_det_inv_jacobian[0].end(),
            local_geometric_quantities_.begin());
  std::copy(
      local_face_det_jacobian[0].begin(), local_face_det_jacobian[0].end(),
      local_geometric_quantities_.begin() +
          static_cast<std::ptrdiff_t>(local_volume_det_inv_jacobian[0].size()));
  std::copy(
      local_face_normal_magnitude[0].begin(),
      local_face_normal_magnitude[0].end(),
      local_geometric_quantities_.begin() +
          static_cast<std::ptrdiff_t>(local_volume_det_inv_jacobian[0].size() +
                                      local_face_det_jacobian[0].size()));
}

template <size_t Dim>
void MortarData<Dim>::insert_local_face_normal_magnitude(
    const Scalar<DataVector>& local_face_normal_magnitude) {
  ASSERT(local_mortar_data_[mortar_index_].has_value(),
         "Must set local mortar data before setting the local face normal.");
  ASSERT(not using_volume_and_face_jacobians_,
         "The face normal magnitude cannot be inserted if the face normal, "
         "volume inverse Jacobian determinant, and face Jacobian determinant "
         "are being used.");
  using_only_face_normal_magnitude_ = true;
  const size_t required_storage_size = local_face_normal_magnitude[0].size();
  local_geometric_quantities_.destructive_resize(required_storage_size);

  std::copy(local_face_normal_magnitude[0].begin(),
            local_face_normal_magnitude[0].end(),
            local_geometric_quantities_.begin());
}

template <size_t Dim>
void MortarData<Dim>::get_local_volume_det_inv_jacobian(
    const gsl::not_null<Scalar<DataVector>*> local_volume_det_inv_jacobian)
    const {
  ASSERT(local_mortar_data_[mortar_index_].has_value(),
         "Must set local mortar data before getting the local volume inverse "
         "Jacobian determinant.");
  ASSERT(
      local_geometric_quantities_.size() >
          2 * std::get<0>(*local_mortar_data()).number_of_grid_points(),
      "Cannot retrieve the volume inverse Jacobian determinant because it was "
      "not inserted.");
  ASSERT(
      using_volume_and_face_jacobians_,
      "Cannot retrieve the volume inverse Jacobian determinant because it was "
      "not inserted.");
  ASSERT(not using_only_face_normal_magnitude_,
         "Inconsistent internal state: we are apparently using both the volume "
         "and face Jacobians, as well as only the face normal.");
  const size_t num_face_points =
      std::get<0>(*local_mortar_data()).number_of_grid_points();
  const size_t num_volume_points =
      local_geometric_quantities_.size() - 2 * num_face_points;
  get(*local_volume_det_inv_jacobian)
      .set_data_ref(const_cast<double*>(  // NOLINT
                        local_geometric_quantities_.data()),
                    num_volume_points);
}

template <size_t Dim>
void MortarData<Dim>::get_local_face_det_jacobian(
    const gsl::not_null<Scalar<DataVector>*> local_face_det_jacobian) const {
  ASSERT(local_mortar_data_[mortar_index_].has_value(),
         "Must set local mortar data before getting the local face Jacobian "
         "determinant.");
  ASSERT(local_geometric_quantities_.size() >
             2 * std::get<0>(*local_mortar_data()).number_of_grid_points(),
         "Cannot retrieve the face Jacobian determinant because it was not "
         "inserted.");
  ASSERT(using_volume_and_face_jacobians_,
         "Cannot retrieve the face Jacobian determinant because it was not "
         "inserted.");
  ASSERT(not using_only_face_normal_magnitude_,
         "Inconsistent internal state: we are apparently using both the volume "
         "and face Jacobians, as well as only the face normal.");
  const size_t num_face_points =
      std::get<0>(*local_mortar_data()).number_of_grid_points();
  const size_t offset =
      local_geometric_quantities_.size() - 2 * num_face_points;
  get(*local_face_det_jacobian)
      .set_data_ref(
          // NOLINTNEXTLINE
          const_cast<double*>(  // NOLINTNEXTLINE
              local_geometric_quantities_.data() + offset),
          num_face_points);
}

template <size_t Dim>
void MortarData<Dim>::get_local_face_normal_magnitude(
    const gsl::not_null<Scalar<DataVector>*> local_face_normal_magnitude)
    const {
  ASSERT(local_mortar_data_[mortar_index_].has_value(),
         "Must set local mortar data before getting the local face normal "
         "magnitude.");
  const size_t num_face_points =
      std::get<0>(*local_mortar_data()).number_of_grid_points();
  ASSERT(local_geometric_quantities_.size() == num_face_points or
             local_geometric_quantities_.size() > 2 * num_face_points,
         "Cannot retrieve the face normal magnitude because it was not "
         "inserted.");
  const size_t offset = local_geometric_quantities_.size() - num_face_points;
  get(*local_face_normal_magnitude)
      .set_data_ref(
          // NOLINTNEXTLINE
          const_cast<double*>(  // NOLINTNEXTLINE
              local_geometric_quantities_.data() + offset),
          num_face_points);
}

template <size_t Dim>
std::pair<std::pair<Mesh<Dim - 1>, DataVector>,
          std::pair<Mesh<Dim - 1>, DataVector>>
MortarData<Dim>::extract() {
  ASSERT(
      local_mortar_data_[mortar_index_].has_value() and
          neighbor_mortar_data_[mortar_index_].has_value(),
      "Tried to extract boundary data, but do not have "
          << (local_mortar_data_[mortar_index_].has_value()
                  ? "neighbor"
                  : neighbor_mortar_data_[mortar_index_].has_value() ? "local"
                                                                     : "any")
          << " data.");
  auto result = std::pair{std::move(*local_mortar_data_[mortar_index_]),
                          std::move(*neighbor_mortar_data_[mortar_index_])};
  local_mortar_data_[mortar_index_].reset();
  neighbor_mortar_data_[mortar_index_].reset();
  return result;
}

template <size_t Dim>
void MortarData<Dim>::next_buffer() {
  mortar_index_ =
      mortar_index_ + 1 == number_of_buffers_ ? 0 : mortar_index_ + 1;
}

template <size_t Dim>
size_t MortarData<Dim>::current_buffer_index() const {
  return mortar_index_;
}

template <size_t Dim>
size_t MortarData<Dim>::total_number_of_buffers() const {
  return number_of_buffers_;
}

template <size_t Dim>
void MortarData<Dim>::pup(PUP::er& p) {
  p | number_of_buffers_;
  p | mortar_index_;
  p | time_step_id_;
  p | local_mortar_data_;
  p | neighbor_mortar_data_;
  p | local_geometric_quantities_;
  p | using_volume_and_face_jacobians_;
  p | using_only_face_normal_magnitude_;
}

template <size_t Dim>
bool operator==(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs) {
  return lhs.number_of_buffers_ == rhs.number_of_buffers_ and
         lhs.mortar_index_ == rhs.mortar_index_ and
         lhs.time_step_id() == rhs.time_step_id() and
         lhs.local_mortar_data() == rhs.local_mortar_data() and
         lhs.neighbor_mortar_data() == rhs.neighbor_mortar_data() and
         lhs.local_geometric_quantities_ == rhs.local_geometric_quantities_ and
         lhs.using_volume_and_face_jacobians_ ==
             rhs.using_volume_and_face_jacobians_ and
         lhs.using_only_face_normal_magnitude_ ==
             rhs.using_only_face_normal_magnitude_;
}

template <size_t Dim>
bool operator!=(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const MortarData<Dim>& mortar_data) {
  os << "TimeStepId: " << mortar_data.time_step_id() << "\n";
  os << "LocalMortarData: " << mortar_data.local_mortar_data() << "\n";
  os << "NeighborMortarData: " << mortar_data.neighbor_mortar_data() << "\n";
  return os;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                \
  template class MortarData<DIM(data)>;                       \
  template bool operator==(const MortarData<DIM(data)>& lhs,  \
                           const MortarData<DIM(data)>& rhs); \
  template bool operator!=(const MortarData<DIM(data)>& lhs,  \
                           const MortarData<DIM(data)>& rhs); \
  template std::ostream& operator<<(std::ostream& os,         \
                                    const MortarData<DIM(data)>& mortar_data);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
