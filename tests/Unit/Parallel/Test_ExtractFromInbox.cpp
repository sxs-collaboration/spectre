// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/ExtractFromInbox.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct TemporalIdTag : db::SimpleTag {
  using type = size_t;
};

struct SampleDataTag {
  using type = std::map<size_t, int>;
};

struct ConstructionObserverTag {
  using type = std::map<size_t, ConstructionObserver>;
};

SPECTRE_TEST_CASE("Unit.Parallel.ExtractFromInbox", "[Parallel][Unit]") {
  const size_t temporal_id = 0;
  const auto box = db::create<db::AddSimpleTags<TemporalIdTag>>(temporal_id);
  tuples::TaggedTuple<SampleDataTag, ConstructionObserverTag> inboxes{};

  tuples::get<SampleDataTag>(inboxes).emplace(temporal_id, 1);
  CHECK(Parallel::extract_from_inbox<SampleDataTag, TemporalIdTag>(inboxes,
                                                                   box) == 1);
  CHECK(tuples::get<SampleDataTag>(inboxes).empty());

  // Make sure no data gets copied it is extracted
  tuples::get<ConstructionObserverTag>(inboxes).emplace(
      std::piecewise_construct, std::make_tuple(temporal_id),
      std::make_tuple());
  CHECK(tuples::get<ConstructionObserverTag>(inboxes).at(temporal_id).status ==
        "initial");
  const auto extracted_observer =
      Parallel::extract_from_inbox<ConstructionObserverTag, TemporalIdTag>(
          inboxes, box);
  CHECK(extracted_observer.status == "move-constructed");
  CHECK(tuples::get<ConstructionObserverTag>(inboxes).empty());
}
}  // namespace
