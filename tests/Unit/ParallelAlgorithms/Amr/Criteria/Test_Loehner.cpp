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
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Loehner.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Criteria {
namespace {

template <size_t Dim>
struct TestVector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        amr::Criterion, tmpl::list<Loehner<Dim, tmpl::list<TestVector<Dim>>>>>>;
  };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Criteria.Loehner", "[Unit][ParallelAlgorithms]") {
  static constexpr size_t Dim = 2;
  const Mesh<Dim> mesh{6, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);

  {
    INFO("Linear function in both dimensions");
    const DataVector test_data =
        get<0>(logical_coords) + 2. * get<1>(logical_coords);
    const auto indicator = loehner_smoothness_indicator(test_data, mesh);
    CHECK(indicator[0] == approx(0.));
    CHECK(indicator[1] == approx(0.));
  }
  {
    INFO("Nonlinear in one dimension and linear in the other");
    const DataVector test_data =
        exp(-square(get<0>(logical_coords))) + 2. * get<1>(logical_coords);
    const auto indicator = loehner_smoothness_indicator(test_data, mesh);
    CHECK(indicator[0] == approx(1.32070910780672479));
    CHECK(indicator[1] == approx(0.));
  }

  register_factory_classes_with_charm<Metavariables<Dim>>();
  const auto criterion = TestHelpers::test_factory_creation<
      amr::Criterion, Loehner<Dim, tmpl::list<TestVector<Dim>>>>(
      "Loehner:\n"
      "  VariablesToMonitor: [TestVector]\n"
      "  AbsoluteTolerance: 1.e-3\n"
      "  RelativeTolerance: 1.e-3\n"
      "  CoarseningFactor: 0.3\n");

  // Manufacture some test data
  tnsr::I<DataVector, Dim> test_data{};
  // X-component is linear in x and y
  get<0>(test_data) = get<0>(logical_coords) + 2. * get<1>(logical_coords);
  // Y-component is nonlinear in one dimension and linear in the other
  get<1>(test_data) =
      exp(-square(get<0>(logical_coords))) + 2. * get<1>(logical_coords);

  Parallel::GlobalCache<Metavariables<Dim>> empty_cache{};
  const auto databox =
      db::create<tmpl::list<::domain::Tags::Mesh<Dim>, TestVector<Dim>>>(
          mesh, std::move(test_data));
  ObservationBox<
      tmpl::list<>,
      db::DataBox<tmpl::list<::domain::Tags::Mesh<Dim>, TestVector<Dim>>>>
      box{databox};

  const auto flags = criterion->evaluate(box, empty_cache, ElementId<Dim>{0});
  CHECK(flags[0] == amr::Flag::Split);
  CHECK(flags[1] == amr::Flag::Join);
}

}  // namespace amr::Criteria
