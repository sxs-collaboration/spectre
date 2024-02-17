// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/IncreaseResolution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Criteria {
namespace {
template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<amr::Criterion, tmpl::list<IncreaseResolution<Dim>>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Criteria.IncreaseResolution",
                  "[Unit][ParallelAlgorithms]") {
  const auto criterion =
      TestHelpers::test_factory_creation<amr::Criterion, IncreaseResolution<2>>(
          "IncreaseResolution");
  Parallel::GlobalCache<Metavariables<2>> empty_cache{};
  auto databox = db::create<tmpl::list<>>();
  ObservationBox<tmpl::list<>, db::DataBox<tmpl::list<>>> box{
      make_not_null(&databox)};
  const ElementId<2> element_id{0};
  const auto flags = criterion->evaluate(box, empty_cache, element_id);
  CHECK(flags == make_array<2>(amr::Flag::IncreaseResolution));
}
}  // namespace amr::Criteria
