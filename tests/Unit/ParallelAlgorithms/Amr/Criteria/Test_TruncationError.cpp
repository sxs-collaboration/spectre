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
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/TruncationError.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

using namespace std::complex_literals;

namespace amr::Criteria {
namespace {

template <typename DataType, size_t Dim>
struct TestVector : db::SimpleTag {
  using type = tnsr::I<DataType, Dim>;
};

template <typename DataType, size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<amr::Criterion,
                             tmpl::list<TruncationError<
                                 Dim, tmpl::list<TestVector<DataType, Dim>>>>>>;
  };
};

template <typename DataType>
void test() {
  static constexpr size_t Dim = 2;
  register_factory_classes_with_charm<Metavariables<DataType, Dim>>();
  const TruncationError<Dim, tmpl::list<TestVector<DataType, Dim>>> criterion{
      {"TestVector"}, 1.e-3, 0.0};
  const auto criterion_from_option_string = TestHelpers::test_factory_creation<
      amr::Criterion,
      TruncationError<Dim, tmpl::list<TestVector<DataType, Dim>>>>(
      "TruncationError:\n"
      "  VariablesToMonitor: [TestVector]\n"
      "  AbsoluteTarget: 1.e-3\n"
      "  RelativeTarget: 0.0\n");

  const auto evaluate_criterion = [&](const size_t num_points) {
    const Mesh<Dim> mesh{num_points, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const auto logical_coords = logical_coordinates(mesh);
    // Manufacture some test data
    tnsr::I<DataType, Dim> test_data{};
    // X-component is linear in x and y, so it is exactly represented on the
    // mesh
    get<0>(test_data) = get<0>(logical_coords) + 2. * get<1>(logical_coords);
    // Y-component is nonlinear in one dimension and linear in the other
    get<1>(test_data) =
        exp(sin(M_PI * get<0>(logical_coords))) + 2. * get<1>(logical_coords);
    if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
      get<0>(test_data) += 1.0i * get<0>(logical_coords);
      get<1>(test_data) += 2.0i * get<1>(logical_coords);
    }

    Parallel::GlobalCache<Metavariables<DataType, Dim>> empty_cache{};
    auto databox = db::create<
        tmpl::list<::domain::Tags::Mesh<Dim>, TestVector<DataType, Dim>>>(
        mesh, std::move(test_data));
    const ObservationBox<tmpl::list<>,
                         db::DataBox<tmpl::list<::domain::Tags::Mesh<Dim>,
                                                TestVector<DataType, Dim>>>>
        box{make_not_null(&databox)};

    const auto result = criterion.evaluate(box, empty_cache, ElementId<Dim>{0});
    CHECK(criterion_from_option_string->evaluate(box, empty_cache,
                                                 ElementId<Dim>{0}) == result);
    return result;
  };

  // Expectation:
  // - In the first dimension, one of the components is nonlinear, so we need
  //   lots of resolution there.
  // - In the second dimension, both components are linear. Technically, we need
  //   only 2 modes to represent the functions exactly. However, if we have only
  //   2 modes we can't be sure numerically that we're resolving the function.
  //   Also with 3 modes we can't be sure, because the third mode might be zero
  //   by symmetry. So we need 4 modes in this dimension.
  {
    INFO("3 modes");
    const auto flags = evaluate_criterion(3);
    CHECK(flags[0] == amr::Flag::IncreaseResolution);
    CHECK(flags[1] == amr::Flag::IncreaseResolution);
  }
  {
    INFO("4 modes");
    const auto flags = evaluate_criterion(4);
    CHECK(flags[0] == amr::Flag::IncreaseResolution);
    CHECK(flags[1] == amr::Flag::DoNothing);
  }
  {
    INFO("5 modes");
    const auto flags = evaluate_criterion(5);
    CHECK(flags[0] == amr::Flag::IncreaseResolution);
    CHECK(flags[1] == amr::Flag::DecreaseResolution);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Criteria.TruncationError",
                  "[Unit][ParallelAlgorithms]") {
  test<DataVector>();
  test<ComplexDataVector>();
}

}  // namespace amr::Criteria
