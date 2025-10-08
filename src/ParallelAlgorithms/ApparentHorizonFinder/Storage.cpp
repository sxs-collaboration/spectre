// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"

#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ah::Storage {
template <typename Fr>
void VolumeVariables<Fr>::pup(PUP::er& p) {
  p | mesh;
  p | vars_to_interpolate_to_target;
}

template <typename Fr>
bool operator==(const VolumeVariables<Fr>& lhs,
                const VolumeVariables<Fr>& rhs) {
  return lhs.mesh == rhs.mesh and
         lhs.vars_to_interpolate_to_target == rhs.vars_to_interpolate_to_target;
}
template <typename Fr>
bool operator!=(const VolumeVariables<Fr>& lhs,
                const VolumeVariables<Fr>& rhs) {
  return not(lhs == rhs);
}

template <typename Fr>
void Iteration<Fr>::reset_for_next_iteration() {
  // Leave the strahlkorper because this was set by FastFlow and is already
  // the next surface
  this->block_coord_holders.reset();
  this->indicies_interpolated_to_thus_far.clear();
  this->interpolation_is_done_for_these_elements.clear();
  this->compute_coords_retries = 0;
}

template <typename Fr>
void Iteration<Fr>::pup(PUP::er& p) {
  p | strahlkorper;
  p | block_coord_holders;
  p | interpolated_vars;
  p | indicies_interpolated_to_thus_far;
  p | interpolation_is_done_for_these_elements;
  p | compute_coords_retries;
}

template <typename Fr>
bool operator==(const Iteration<Fr>& lhs, const Iteration<Fr>& rhs) {
  return lhs.strahlkorper == rhs.strahlkorper and
         lhs.block_coord_holders == rhs.block_coord_holders and
         lhs.interpolated_vars == rhs.interpolated_vars and
         lhs.indicies_interpolated_to_thus_far ==
             rhs.indicies_interpolated_to_thus_far and
         lhs.interpolation_is_done_for_these_elements ==
             rhs.interpolation_is_done_for_these_elements and
         lhs.compute_coords_retries == rhs.compute_coords_retries;
}
template <typename Fr>
bool operator!=(const Iteration<Fr>& lhs, const Iteration<Fr>& rhs) {
  return not(lhs == rhs);
}

template <typename Fr>
void SingleTimeStorage<Fr>::pup(PUP::er& p) {
  p | all_volume_variables;
  p | current_iteration;
  p | previous_iteration_surface;
  p | destination;
  p | time_is_ready;
}

template <typename Fr>
bool operator==(const SingleTimeStorage<Fr>& lhs,
                const SingleTimeStorage<Fr>& rhs) {
  return lhs.all_volume_variables == rhs.all_volume_variables and
         lhs.current_iteration == rhs.current_iteration and
         lhs.previous_iteration_surface == rhs.previous_iteration_surface and
         lhs.destination == rhs.destination and
         lhs.time_is_ready == rhs.time_is_ready;
}
template <typename Fr>
bool operator!=(const SingleTimeStorage<Fr>& lhs,
                const SingleTimeStorage<Fr>& rhs) {
  return not(lhs == rhs);
}

template <typename Fr>
PreviousSurface<Fr>::PreviousSurface(const LinkedMessageId<double>& time_in,
                                     ylm::Strahlkorper<Fr> surface_in)
    : time(time_in), surface(std::move(surface_in)) {}

template <typename Fr>
void PreviousSurface<Fr>::pup(PUP::er& p) {
  p | time;
  p | surface;
}

template <typename Fr>
bool operator==(const PreviousSurface<Fr>& lhs,
                const PreviousSurface<Fr>& rhs) {
  return lhs.time == rhs.time and lhs.surface == rhs.surface;
}
template <typename Fr>
bool operator!=(const PreviousSurface<Fr>& lhs,
                const PreviousSurface<Fr>& rhs) {
  return not(lhs == rhs);
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template struct VolumeVariables<FRAME(data)>;                    \
  template struct Iteration<FRAME(data)>;                          \
  template struct SingleTimeStorage<FRAME(data)>;                  \
  template struct PreviousSurface<FRAME(data)>;                    \
  template bool operator==(const VolumeVariables<FRAME(data)>&,    \
                           const VolumeVariables<FRAME(data)>&);   \
  template bool operator!=(const VolumeVariables<FRAME(data)>&,    \
                           const VolumeVariables<FRAME(data)>&);   \
  template bool operator==(const Iteration<FRAME(data)>&,          \
                           const Iteration<FRAME(data)>&);         \
  template bool operator!=(const Iteration<FRAME(data)>&,          \
                           const Iteration<FRAME(data)>&);         \
  template bool operator==(const SingleTimeStorage<FRAME(data)>&,  \
                           const SingleTimeStorage<FRAME(data)>&); \
  template bool operator!=(const SingleTimeStorage<FRAME(data)>&,  \
                           const SingleTimeStorage<FRAME(data)>&); \
  template bool operator==(const PreviousSurface<FRAME(data)>&,    \
                           const PreviousSurface<FRAME(data)>&);   \
  template bool operator!=(const PreviousSurface<FRAME(data)>&,    \
                           const PreviousSurface<FRAME(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Inertial, Frame::Distorted, Frame::Grid))

#undef INSTANTIATE
#undef FRAME
}  // namespace ah::Storage
