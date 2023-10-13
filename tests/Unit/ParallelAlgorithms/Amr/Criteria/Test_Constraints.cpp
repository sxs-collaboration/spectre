// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Constraints.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Criteria {
namespace {

template <size_t Dim>
struct TestVector : db::SimpleTag {
  using type = tnsr::ia<DataVector, Dim>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<amr::Criterion,
                   tmpl::list<Constraints<Dim, tmpl::list<TestVector<Dim>>>>>>;
  };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Criteria.Constraints",
                  "[Unit][ParallelAlgorithms]") {
  static constexpr size_t Dim = 2;
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  register_factory_classes_with_charm<Metavariables<Dim>>();
  const auto criterion = TestHelpers::test_factory_creation<
      amr::Criterion, Constraints<Dim, tmpl::list<TestVector<Dim>>>>(
      "Constraints:\n"
      "  ConstraintsToMonitor: [TestVector]\n"
      "  AbsoluteTarget: 1.e-3\n"
      "  CoarseningFactor: 0.1\n");

  // Set up a grid
  const Mesh<Dim> mesh{4, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);
  // Logical to inertial coords is a rotation by pi/2 so the directions are
  // swapped
  const auto map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine2D{Affine{-1., 1., -1., 1.}, Affine{-1., 1., -1., 1.}},
          domain::CoordinateMaps::DiscreteRotation(OrientationMap<Dim>{
              {{Direction<Dim>::lower_eta(), Direction<Dim>::upper_xi()}}}));
  auto jacobian = map.jacobian(logical_coords);

  // Manufacture some test data
  tnsr::ia<DataVector, Dim> test_data{mesh.number_of_grid_points()};
  // X-component (logical eta direction) is small
  get<0, 0>(test_data) = 1.e-6;
  get<0, 1>(test_data) = 1.e-7;
  get<0, 2>(test_data) = 1.e-8;
  // Y-component (logical xi direction) is large
  get<1, 0>(test_data) = 1.e-1;
  get<1, 1>(test_data) = 1.e-2;
  get<1, 2>(test_data) = 1.e-3;

  Parallel::GlobalCache<Metavariables<Dim>> empty_cache{};
  auto databox =
      db::create<tmpl::list<Events::Tags::ObserverJacobian<
                                Dim, Frame::ElementLogical, Frame::Inertial>,
                            TestVector<Dim>>>(std::move(jacobian),
                                              std::move(test_data));
  ObservationBox<
      tmpl::list<>,
      db::DataBox<tmpl::list<Events::Tags::ObserverJacobian<
                                 Dim, Frame::ElementLogical, Frame::Inertial>,
                             TestVector<Dim>>>>
      box{make_not_null(&databox)};

  const auto flags = criterion->evaluate(box, empty_cache, ElementId<Dim>{0});
  CHECK(flags[0] == amr::Flag::IncreaseResolution);
  CHECK(flags[1] == amr::Flag::DecreaseResolution);
}

}  // namespace amr::Criteria
