// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file is tested in Test_InboxTags.cpp

#pragma once

#include <atomic>
#include <boost/container/small_vector.hpp>
#include <cstddef>
#include <map>
#include <tuple>
#include <utility>

#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Parallel/StaticSpscQueue.hpp"
#include "Time/TimeStepId.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace evolution::dg {
/*!
 * \brief Holds the data in the different directions for the nodegroup
 * `DgElementArray` implementation.
 *
 * The reason for this class is to reduce contention between cores and to
 * allow the use of a Single-Producer-Single-Consumer (SPSC) queue instead of
 * an MPMC queue. This has significant performance improvements since it
 * drastically reduces contention.
 *
 * \warning Only `AtomicInboxBoundaryData` with zero messages can be
 * move constructed or serialized.  This is necessary in order to be
 * able to serialize a
 * `std::unordered_map<Key,AtomicInboxBoundaryData>`.
 */
template <size_t Dim>
struct AtomicInboxBoundaryData {
  using stored_type = evolution::dg::BoundaryData<Dim>;
  AtomicInboxBoundaryData() = default;
  AtomicInboxBoundaryData(const AtomicInboxBoundaryData&) = delete;
  AtomicInboxBoundaryData& operator=(const AtomicInboxBoundaryData&) = delete;
  AtomicInboxBoundaryData(AtomicInboxBoundaryData&& rhs) noexcept;
  AtomicInboxBoundaryData& operator=(AtomicInboxBoundaryData&&) noexcept =
      delete;
  ~AtomicInboxBoundaryData() = default;

  /*!
   * Computes the 1d index into the `boundary_data_in_directions` array
   * for a specific `directional_id` that has been re-oriented using the
   * `OrientationMap` to be put in the same block frame as the element that is
   * receiving the data (i.e. that whose inbox this is being inserted into).
   *
   * The hash is computed as
   * \f{align}{
   * 2^D d + 2^{D-1} s + e
   * \f}
   * where \f$D\f$ is the number of spatial dimensions, \f$d\f$ is the logical
   * dimension of the direction to the neighbor from the element whose inbox
   * this is, \f$s\f$ is the side in the logical dimension \f$d\f$ with a value
   * of 1 for upper and 0 for lower, and \f$e\f$ is a hash of the index of the
   * `SegmentId`'s of the neighbor's `ElementId` for the dimensions other than
   * \f$d\f$. In particular: for \f$d=1\f$, \f$e\f$ is 0 (1) if
   * the `SegmentId` index along the face is even (odd); and for \f$d = 3\f$
   * \f$e\f$ is 0 (1, 2, 3) if the `SegmentId` indices along the face are both
   * even (lower dim odd, higher dim odd, both dims odd). The element segment
   * hash is computed as the logical `and` of the `SegmentID`'s index in that
   * direction, left shifted by which direction on the face it is.
   */
  static size_t index(const DirectionalId<Dim>& directional_id);

  /*!
   * Moves data from the SPSC queues into the `messages` map.
   */
  void collect_messages();

  /*!
   * Set a lower bound on the number of messages required for the
   * algorithm to make progress since the most recent call to
   * `collect_messages`.  After that number of new messages have been
   * received, `BoundaryCorrectionAndGhostCellsInbox` will restart
   * the algorithm.
   *
   * \return whether enough messages have been received.
   */
  bool set_missing_messages(size_t count);

  void pup(PUP::er& p);

  // We use 20 entries in the SPSC under the assumption that each neighbor
  // will never insert more than 20 entries before the element uses
  // them. While in practice a smaller buffer could be used, this is to
  // safeguard against future features.
  std::array<Parallel::StaticSpscQueue<
                 std::tuple<::TimeStepId, stored_type, DirectionalId<Dim>>, 20>,
             maximum_number_of_neighbors(Dim)>
      boundary_data_in_directions{};
  std::map<TimeStepId, boost::container::small_vector<
                           std::pair<DirectionalId<Dim>, stored_type>,
                           maximum_number_of_neighbors(Dim)>>
      messages{};
  std::atomic_int missing_messages{};

  // The number of messages in the SPSC queues is
  // passed_missing_messages - missing_messages - processed_messages.
  // The various manipulations change these fields as follows:
  //
  // New message => --missing_messages (by insert_into_inbox)
  // collect_messages => processed_messages += num unqueued messages
  // set_missing_messages(count) =>
  //   missing_messages += count + processed_messages - passed_missing_messages
  //   processed_messages = 0
  //   passed_missing_messages = count
  //
  // The processed_messages field exists (rather than just including
  // it in missing_messages) so that missing_messages is positive if
  // and only if the element is blocked on messages in this queue.
  int processed_messages{};
  int passed_missing_messages{};
};

/// \brief `std::true` if `T` is a `AtomicInboxBoundaryData`
template <typename T>
struct is_atomic_inbox_boundary_data : std::false_type {};

/// \cond
template <size_t Dim>
struct is_atomic_inbox_boundary_data<AtomicInboxBoundaryData<Dim>>
    : std::true_type {};
/// \endcond

/// \brief `true` if `T` is a `AtomicInboxBoundaryData`
template <typename T>
constexpr size_t is_atomic_inbox_boundary_data_v =
    is_atomic_inbox_boundary_data<T>::value;
}  // namespace evolution::dg
