// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/IncreaseResolution.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::Criteria {
namespace {
template <typename Frame>
struct Metavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<ah::Criterion, tmpl::list<IncreaseResolution>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.Criteria.IncreaseResolution",
                  "[Unit][ParallelAlgorithms]") {
  TestHelpers::db::test_simple_tag<ah::Criteria::Tags::Criteria>("Criteria");
  const auto criterion =
      TestHelpers::test_factory_creation<ah::Criterion, IncreaseResolution>(
          "IncreaseResolution");
  Parallel::GlobalCache<Metavariables<Frame::Inertial>> empty_cache{};
  auto databox = db::create<tmpl::list<>>();
  const ObservationBox<tmpl::list<>, db::DataBox<tmpl::list<>>> box{
      make_not_null(&databox)};

  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{
      11, 4.5, {{0.1, 0.2, 0.3}}};
  const FastFlow::IterInfo iter_info{};

  const size_t new_l_max =
      criterion->evaluate(box, empty_cache, strahlkorper, iter_info);
  CHECK(new_l_max == strahlkorper.l_max() + 2);
}
}  // namespace ah::Criteria
