// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/Neighbors.hpp"

#include <algorithm>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>

#include "Domain/Structure/ElementId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
template <size_t VolumeDim>
bool ids_and_orientations_are_consistent(
    const std::unordered_set<size_t>& block_ids,
    const std::unordered_map<size_t, OrientationMap<VolumeDim>>& orientations) {
  return alg::none_of(block_ids, [&orientations](size_t block_id) {
    return orientations.count(block_id) == 0;
  });
}

template <size_t VolumeDim>
bool ids_and_orientations_are_consistent(
    const std::unordered_set<ElementId<VolumeDim>>& element_ids,
    const std::unordered_map<size_t, OrientationMap<VolumeDim>>& orientations) {
  return alg::none_of(element_ids,
                      [&orientations](const ElementId<VolumeDim>& element_id) {
                        return orientations.count(element_id.block_id()) == 0;
                      });
}
}  // namespace

template <size_t VolumeDim, typename IdType>
Neighbors<VolumeDim, IdType>::Neighbors(
    std::unordered_set<IdType> ids,
    std::unordered_map<size_t, OrientationMap<VolumeDim>> orientations,
    const bool are_conforming)
    : ids_(std::move(ids)),
      orientations_(std::move(orientations)),
      are_conforming_(are_conforming) {
  ASSERT(alg::none_of(orientations_,
                      [](const auto& orientation) {
                        return orientation.second ==
                               OrientationMap<VolumeDim>{};
                      }),
         "Cannot use a default-constructed OrientationMap in Neighbors.");
  ASSERT(orientations_.size() == 1 or not are_conforming_,
         "Conforming neighbors must all be in the same block, but multiple "
         "orientations were specified "
             << orientations_);
  ASSERT(ids_and_orientations_are_consistent(ids_, orientations_),
         "Not all Ids " << ids_ << " have orientations " << orientations_);
}

template <size_t VolumeDim, typename IdType>
Neighbors<VolumeDim, IdType>::Neighbors(std::unordered_set<IdType> ids,
                                        OrientationMap<VolumeDim> orientation)
    : ids_(std::move(ids)) {
  ASSERT(orientation != OrientationMap<VolumeDim>{},
         "Cannot use a default-constructed OrientationMap in Neighbors.");
  if constexpr (std::is_same_v<size_t, IdType>) {
    orientations_.emplace(*ids_.begin(), std::move(orientation));
  } else {
    orientations_.emplace(ids_.begin()->block_id(), std::move(orientation));
  }
  ASSERT(ids_and_orientations_are_consistent(ids_, orientations_),
         "Conforming neighbors must all be in the same block, but the "
         "following Ids were specified: "
             << ids_);
}

template <size_t VolumeDim, typename IdType>
Neighbors<VolumeDim, IdType>::Neighbors(const IdType id,
                                        OrientationMap<VolumeDim> orientation)
    : Neighbors(std::unordered_set{std::move(id)}, std::move(orientation)) {}

template <size_t VolumeDim, typename IdType>
const OrientationMap<VolumeDim>& Neighbors<VolumeDim, IdType>::orientation(
    const IdType& id) const {
  if constexpr (std::is_same_v<size_t, IdType>) {
    return orientations_.at(id);
  } else {
    return orientations_.at(id.block_id());
  }
}

template <size_t VolumeDim, typename IdType>
void Neighbors<VolumeDim, IdType>::set_ids_to(
    const std::unordered_set<IdType> new_ids) {
  ids_ = std::move(new_ids);
  ASSERT(ids_and_orientations_are_consistent(ids_, orientations_),
         "Some of the Ids " << ids_
                            << "passed to set_ids_to are from different blocks "
                               "than the current orientations "
                            << orientations_);
}

template <size_t VolumeDim, typename IdType>
void Neighbors<VolumeDim, IdType>::add_ids(
    std::unordered_set<IdType> additional_ids) {
  ids_.merge(additional_ids);
  ASSERT(ids_and_orientations_are_consistent(ids_, orientations_),
         "Some of the added Ids "
             << additional_ids
             << " are from different blocks than the current orientations "
             << orientations_);
}

template <size_t VolumeDim, typename IdType>
std::ostream& operator<<(std::ostream& os,
                         const Neighbors<VolumeDim, IdType>& neighbors) {
  os << "Ids = " << neighbors.ids()
     << "; orientations = " << neighbors.orientations()
     << "; conforming = " << std::boolalpha << neighbors.are_conforming();
  return os;
}

template <size_t VolumeDim, typename IdType>
bool operator==(const Neighbors<VolumeDim, IdType>& lhs,
                const Neighbors<VolumeDim, IdType>& rhs) {
  return (lhs.ids() == rhs.ids() and
          lhs.orientations() == rhs.orientations() and
          lhs.are_conforming() == rhs.are_conforming());
}

template <size_t VolumeDim, typename IdType>
bool operator!=(const Neighbors<VolumeDim, IdType>& lhs,
                const Neighbors<VolumeDim, IdType>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim, typename IdType>
void Neighbors<VolumeDim, IdType>::pup(PUP::er& p) {
  if constexpr (std::is_same_v<IdType, size_t>) {
    size_t version = 2;
    p | version;
    if (version == 0) {
      // Deserialize old BlockNeighbor class
      size_t id = 0;
      p | id;
      ids_.clear();
      ids_.emplace(id);
      OrientationMap<VolumeDim> orientation;
      p | orientation;
      orientations_.clear();
      orientations_.emplace(id, orientation);
      are_conforming_ = true;
    } else if (version == 1) {
      p | ids_;
      OrientationMap<VolumeDim> orientation;
      p | orientation;
      orientations_.clear();
      orientations_.emplace(*(ids_.begin()), orientation);
      are_conforming_ = true;
    } else {
      ASSERT(version == 2, "Unknonwn version " << version);
      p | ids_;
      p | orientations_;
      p | are_conforming_;
    }
  } else {
    p | ids_;
    p | orientations_;
    p | are_conforming_;
  }
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template class Neighbors<GET_DIM(data), size_t>;                          \
  template std::ostream& operator<<(                                        \
      std::ostream& os, const Neighbors<GET_DIM(data), size_t>& neighbors); \
  template bool operator==(const Neighbors<GET_DIM(data), size_t>& lhs,     \
                           const Neighbors<GET_DIM(data), size_t>& rhs);    \
  template bool operator!=(const Neighbors<GET_DIM(data), size_t>& lhs,     \
                           const Neighbors<GET_DIM(data), size_t>& rhs);    \
  template class Neighbors<GET_DIM(data)>;                                  \
  template std::ostream& operator<<(                                        \
      std::ostream& os, const Neighbors<GET_DIM(data)>& neighbors);         \
  template bool operator==(const Neighbors<GET_DIM(data)>& lhs,             \
                           const Neighbors<GET_DIM(data)>& rhs);            \
  template bool operator!=(const Neighbors<GET_DIM(data)>& lhs,             \
                           const Neighbors<GET_DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
