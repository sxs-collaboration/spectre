// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file is tested in Test_InboxTags.cpp

#pragma once

#include <cstddef>
#include <map>

#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Time/TimeStepId.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace evolution::dg {
/// Class wrapping a map and mirroring the AtomicInboxBoundaryData
/// interface so that code accessing the inbox doesn't need to care
/// which implementation is in use.
template <size_t Dim>
struct InboxBoundaryData {
  using mapped_type = DirectionalIdMap<Dim, evolution::dg::BoundaryData<Dim>>;

  std::map<TimeStepId, mapped_type> messages;

  /// Number of messages needed to restart the algorithm.  This should
  /// be decremented whenever a new message is added to the inbox.
  int missing_messages = 0;

  bool empty() const;

  /// In AtomicInboxBoundaryData, this moves elements from the
  /// threadsafe structure to the `messages` field.  This class stores
  /// messages in the `messages` field directly, so this method just
  /// zeros the missing message count.
  void collect_messages();

  /// Set a lower bound on the number of messages required for the
  /// algorithm to make progress since the most recent call to
  /// `collect_messages`.  After that number of new messages have been
  /// received, `BoundaryCorrectionAndGhostCellsInbox` will restart
  /// the algorithm.
  ///
  /// \return whether enough messages have been received.
  bool set_missing_messages(size_t count);

  void pup(PUP::er& p);
};
}  // namespace evolution::dg
