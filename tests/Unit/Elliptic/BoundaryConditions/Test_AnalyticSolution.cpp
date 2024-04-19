// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::BoundaryConditions {

namespace {

template <size_t N>
struct ScalarFieldTag : db::SimpleTag {
  static std::string name() { return "Field" + std::to_string(N); }
  using type = Scalar<DataVector>;
};

template <size_t Dim, size_t N>
struct FluxTag : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using primal_fields = tmpl::list<ScalarFieldTag<1>, ScalarFieldTag<2>>;
  using primal_fluxes = tmpl::list<FluxTag<Dim, 1>, FluxTag<Dim, 2>>;
};

template <size_t Dim>
using test_tags = tmpl::list<ScalarFieldTag<1>, ScalarFieldTag<2>,
                             FluxTag<Dim, 1>, FluxTag<Dim, 2>>;

template <size_t Dim>
struct TestSolution : elliptic::analytic_data::AnalyticSolution {
  using options = tmpl::list<>;
  static constexpr Options::String help{"A solution."};
  TestSolution() = default;
  TestSolution(const TestSolution&) = default;
  TestSolution& operator=(const TestSolution&) = default;
  TestSolution(TestSolution&&) = default;
  TestSolution& operator=(TestSolution&&) = default;
  ~TestSolution() override = default;
  explicit TestSolution(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TestSolution);  // NOLINT

  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<TestSolution>(*this);
  }

  tuples::tagged_tuple_from_typelist<test_tags<Dim>> variables(
      const tnsr::I<DataVector, Dim>& x, test_tags<Dim> /*meta*/) const {
    // Create arbitrary analytic solution data
    Variables<test_tags<Dim>> result{x.begin()->size()};
    std::iota(result.data(), result.data() + result.size(), 1.);
    return {get<ScalarFieldTag<1>>(result), get<ScalarFieldTag<2>>(result),
            get<FluxTag<Dim, 1>>(result), get<FluxTag<Dim, 2>>(result)};
  }
};

template <size_t Dim>
PUP::able::PUP_ID TestSolution<Dim>::my_PUP_ID = 0;  // NOLINT

template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<elliptic::BoundaryConditions::BoundaryCondition<Dim>,
                   tmpl::list<elliptic::BoundaryConditions::AnalyticSolution<
                       System<Dim>>>>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                   tmpl::list<TestSolution<Dim>>>>;
  };
};

template <size_t Dim>
void test_analytic_solution() {
  CAPTURE(Dim);
  // Test factory-creation
  register_factory_classes_with_charm<Metavariables<Dim>>();
  const auto created = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<BoundaryCondition<Dim>>,
                                 Metavariables<Dim>>(
          "AnalyticSolution:\n"
          "  Solution: TestSolution\n"
          "  Field1: Dirichlet\n"
          "  Field2: Neumann"));
  REQUIRE(dynamic_cast<const AnalyticSolution<System<Dim>>*>(created.get()) !=
          nullptr);
  const auto& boundary_condition =
      dynamic_cast<const AnalyticSolution<System<Dim>>&>(*created);
  {
    INFO("Properties");
    CHECK(boundary_condition.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              elliptic::BoundaryConditionType::Dirichlet,
              elliptic::BoundaryConditionType::Neumann});
  }
  {
    INFO("Test applying the boundary conditions");
    const Mesh<Dim> volume_mesh{3, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
    const auto direction = Direction<Dim>::lower_xi();
    const auto face_mesh = volume_mesh.slice_away(direction.dimension());
    const size_t face_num_points = face_mesh.number_of_grid_points();
    tnsr::I<DataVector, Dim> face_inertial_coords{face_num_points, 0.};
    tnsr::i<DataVector, Dim> face_normal{face_num_points, 0.};
    get<0>(face_normal) = -2.;
    const auto box = db::create<db::AddSimpleTags<
        Parallel::Tags::MetavariablesImpl<Metavariables<Dim>>,
        domain::Tags::Faces<Dim,
                            domain::Tags::Coordinates<Dim, Frame::Inertial>>,
        domain::Tags::Faces<Dim, domain::Tags::FaceNormal<Dim>>>>(
        Metavariables<Dim>{},
        DirectionMap<Dim, tnsr::I<DataVector, Dim>>{
            {direction, face_inertial_coords}},
        DirectionMap<Dim, tnsr::i<DataVector, Dim>>{{direction, face_normal}});
    Variables<tmpl::list<ScalarFieldTag<1>, ScalarFieldTag<2>,
                         ::Tags::NormalDotFlux<ScalarFieldTag<1>>,
                         ::Tags::NormalDotFlux<ScalarFieldTag<2>>>>
        vars{face_num_points, std::numeric_limits<double>::max()};
    const tnsr::i<DataVector, Dim> deriv_scalar{
        face_num_points, std::numeric_limits<double>::signaling_NaN()};
    // Inhomogeneous boundary conditions
    elliptic::apply_boundary_condition<
        false, void, tmpl::list<AnalyticSolution<System<Dim>>>>(
        boundary_condition, box, direction,
        make_not_null(&get<ScalarFieldTag<1>>(vars)),
        make_not_null(&get<ScalarFieldTag<2>>(vars)),
        make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<1>>>(vars)),
        make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<2>>>(vars)),
        deriv_scalar, deriv_scalar);
    // Imposed Dirichlet conditions on field 1
    const auto expected_dirichlet_field = []() -> DataVector {
      if constexpr (Dim == 1) {
        return {1.};
      } else if constexpr (Dim == 2) {
        return {1., 2., 3.};
      } else if constexpr (Dim == 3) {
        return {1., 2., 3., 4., 5., 6., 7., 8., 9.};
      }
    }();
    CHECK_ITERABLE_APPROX(get(get<ScalarFieldTag<1>>(vars)),
                          expected_dirichlet_field);
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::NormalDotFlux<ScalarFieldTag<1>>>(vars)),
        SINGLE_ARG(
            DataVector{face_num_points, std::numeric_limits<double>::max()}));
    // Imposed Neumann conditions on field 2
    const auto expected_neumann_field = []() -> DataVector {
      if constexpr (Dim == 1) {
        return {-8.};
      } else if constexpr (Dim == 2) {
        return {-26., -28., -30.};
      } else if constexpr (Dim == 3) {
        return {-92., -94., -96., -98., -100., -102., -104., -106., -108.};
      }
    }();
    CHECK_ITERABLE_APPROX(
        get(get<ScalarFieldTag<2>>(vars)),
        SINGLE_ARG(
            DataVector{face_num_points, std::numeric_limits<double>::max()}));
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::NormalDotFlux<ScalarFieldTag<2>>>(vars)),
        expected_neumann_field);
    // Homogeneous (linearized) boundary conditions
    elliptic::apply_boundary_condition<
        true, void, tmpl::list<AnalyticSolution<System<Dim>>>>(
        boundary_condition, box, direction,
        make_not_null(&get<ScalarFieldTag<1>>(vars)),
        make_not_null(&get<ScalarFieldTag<2>>(vars)),
        make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<1>>>(vars)),
        make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<2>>>(vars)),
        deriv_scalar, deriv_scalar);
    // Imposed Dirichlet conditions on field 1
    CHECK_ITERABLE_APPROX(get(get<ScalarFieldTag<1>>(vars)),
                          SINGLE_ARG(DataVector{face_num_points, 0.}));
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::NormalDotFlux<ScalarFieldTag<1>>>(vars)),
        SINGLE_ARG(
            DataVector{face_num_points, std::numeric_limits<double>::max()}));
    // Imposed Neumann conditions on field 2
    CHECK_ITERABLE_APPROX(
        get(get<ScalarFieldTag<2>>(vars)),
        SINGLE_ARG(
            DataVector{face_num_points, std::numeric_limits<double>::max()}));
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::NormalDotFlux<ScalarFieldTag<2>>>(vars)),
        SINGLE_ARG(DataVector{face_num_points, 0.}));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.BoundaryConditions.AnalyticSolution",
                  "[Unit][Elliptic]") {
  test_analytic_solution<1>();
  test_analytic_solution<2>();
  test_analytic_solution<3>();
}

}  // namespace elliptic::BoundaryConditions
