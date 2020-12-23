// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"

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
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg {
template <size_t Dim>
void MortarData<Dim>::insert_local_mortar_data(
    TimeStepId time_step_id, Mesh<Dim - 1> local_interface_mesh,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::vector<double> local_mortar_vars) noexcept {
  // clang-tidy can't figure out that `vars` is moved below
  ASSERT(not local_mortar_data_, "Already received local data at "
                                     << time_step_id << " with interface mesh "
                                     << local_interface_mesh);
  ASSERT(not neighbor_mortar_data_ or time_step_id == time_step_id_,
         "Received local data at " << time_step_id
                                   << ", but already have neighbor data at "
                                   << time_step_id_);
  // NOLINTNEXTLINE(performance-move-const-arg)
  time_step_id_ = std::move(time_step_id);
  local_mortar_data_ =
      std::pair{std::move(local_interface_mesh), std::move(local_mortar_vars)};
}

template <size_t Dim>
void MortarData<Dim>::insert_neighbor_mortar_data(
    TimeStepId time_step_id, Mesh<Dim - 1> neighbor_interface_mesh,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::vector<double> neighbor_mortar_vars) noexcept {
  // clang-tidy can't figure out that `vars` is moved below
  ASSERT(not neighbor_mortar_data_, "Already received neighbor data at "
                                        << time_step_id
                                        << " with interface mesh "
                                        << neighbor_interface_mesh);
  ASSERT(not local_mortar_data_ or time_step_id == time_step_id_,
         "Received neighbor data at " << time_step_id
                                      << ", but already have local data at "
                                      << time_step_id_);
  // NOLINTNEXTLINE(performance-move-const-arg)
  time_step_id_ = std::move(time_step_id);
  neighbor_mortar_data_ = std::pair{std::move(neighbor_interface_mesh),
                                    std::move(neighbor_mortar_vars)};
}

template <size_t Dim>
std::pair<std::pair<Mesh<Dim - 1>, std::vector<double>>,
          std::pair<Mesh<Dim - 1>, std::vector<double>>>
MortarData<Dim>::extract() noexcept {
  ASSERT(local_mortar_data_ and neighbor_mortar_data_,
         "Tried to extract boundary data, but do not have "
             << (local_mortar_data_      ? "neighbor"
                 : neighbor_mortar_data_ ? "local"
                                         : "any")
             << " data.");
  auto result = std::pair{std::move(*local_mortar_data_),
                          std::move(*neighbor_mortar_data_)};
  local_mortar_data_.reset();
  neighbor_mortar_data_.reset();
  return result;
}

template <size_t Dim>
void MortarData<Dim>::pup(PUP::er& p) noexcept {
  p | time_step_id_;
  p | local_mortar_data_;
  p | neighbor_mortar_data_;
}

template <size_t Dim>
bool operator==(const MortarData<Dim>& lhs,
                const MortarData<Dim>& rhs) noexcept {
  return lhs.time_step_id() == rhs.time_step_id() and
         lhs.local_mortar_data() == rhs.local_mortar_data() and
         lhs.neighbor_mortar_data() == rhs.neighbor_mortar_data();
}

template <size_t Dim>
bool operator!=(const MortarData<Dim>& lhs,
                const MortarData<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                         \
  template class MortarData<DIM(data)>;                                \
  template bool operator==(const MortarData<DIM(data)>& lhs,           \
                           const MortarData<DIM(data)>& rhs) noexcept; \
  template bool operator!=(const MortarData<DIM(data)>& lhs,           \
                           const MortarData<DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
