// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <cstddef>

#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Parallel/StaticSpscQueue.hpp"
#include "Time/TimeStepId.hpp"

/// \cond
template <size_t Dim>
struct DirectionalId;
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
 * The uint message counter is used to count how many neighbors have contributed
 * for the next time. This is used to delay calling `perform_algorithm()` in
 * order to reduce the number of messages we send through the runtime
 * system. The `number_of_neighbors` is used to track the number of expected
 * messages. Note that some additional logic is needed also for supporting
 * local time stepping, since not every message entry "counts" since it
 * depends on the time level of neighboring elements.
 */
template <size_t Dim>
struct AtomicInboxBoundaryData {
  using stored_type = evolution::dg::BoundaryData<Dim>;

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

  // We use 20 entiries in the SPSC under the assumption that each neighbor
  // will never insert more than 20 entries before the element uses
  // them. While in practice a smaller buffer could be used, this is to
  // safeguard against future features.
  std::array<Parallel::StaticSpscQueue<
                 std::tuple<::TimeStepId, stored_type, DirectionalId<Dim>>, 20>,
             maximum_number_of_neighbors(Dim)>
      boundary_data_in_directions{};
  std::atomic_uint message_count{};
  std::atomic_uint number_of_neighbors{};
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
