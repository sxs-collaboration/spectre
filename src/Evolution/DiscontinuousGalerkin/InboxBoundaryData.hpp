// Distributed under the MIT License.
// See LICENSE.txt for details.

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

  bool empty() const;

  /// In AtomicInboxBoundaryData, this moves elements from the
  /// threadsafe structure to the `messages` field.  This class stores
  /// messages in the `messages` field directly, so this method does nothing.
  void collect_messages();

  void pup(PUP::er& p);
};
}  // namespace evolution::dg
