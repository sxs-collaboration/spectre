// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::BoundaryConditions {

namespace {

template <size_t N>
struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "Field" + std::to_string(N); }
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
void test_analytic_solution() noexcept {
  CAPTURE(Dim);
  // Test factory-creation
  using Registrar = Registrars::AnalyticSolution<System<Dim>>;
  const auto boundary_condition_base = TestHelpers::test_creation<
      std::unique_ptr<BoundaryCondition<Dim, tmpl::list<Registrar>>>>(
      "AnalyticSolution:\n"
      "  Field1: Dirichlet\n"
      "  Field2: Neumann");
  using BoundaryConditionType =
      typename Registrar::template f<tmpl::list<Registrar>>;
  REQUIRE(dynamic_cast<const BoundaryConditionType*>(
              boundary_condition_base.get()) != nullptr);
  const auto& boundary_condition =
      dynamic_cast<const BoundaryConditionType&>(*boundary_condition_base);
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
  {
    INFO("Test applying the boundary conditions");
    const Mesh<Dim> volume_mesh{3, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
    const size_t volume_num_points = volume_mesh.number_of_grid_points();
    const auto direction = Direction<Dim>::lower_xi();
    const auto face_mesh = volume_mesh.slice_away(direction.dimension());
    const size_t face_num_points = face_mesh.number_of_grid_points();
    tnsr::i<DataVector, Dim> face_normal{face_num_points, 0.};
    get<0>(face_normal) = -2.;
    // Create arbitrary analytic solution data
    Variables<tmpl::list<::Tags::Analytic<ScalarFieldTag<1>>,
                         ::Tags::Analytic<ScalarFieldTag<2>>,
                         ::Tags::Analytic<FluxTag<Dim, 1>>,
                         ::Tags::Analytic<FluxTag<Dim, 2>>>>
        analytic_solutions{volume_num_points};
    std::iota(analytic_solutions.data(),
              analytic_solutions.data() + analytic_solutions.size(), 1.);
    const auto box = db::create<db::AddSimpleTags<
        domain::Tags::Mesh<Dim>,
        ::Tags::AnalyticSolutions<
            tmpl::list<ScalarFieldTag<1>, ScalarFieldTag<2>, FluxTag<Dim, 1>,
                       FluxTag<Dim, 2>>>,
        domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
                                domain::Tags::Direction<Dim>>,
        domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>,
            ::Tags::Normalized<
                domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>>>(
        volume_mesh, std::move(analytic_solutions),
        std::unordered_map<Direction<Dim>, Direction<Dim>>{
            {direction, direction}},
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>{
            {direction, face_normal}});
    Variables<tmpl::list<ScalarFieldTag<1>, ScalarFieldTag<2>,
                         ::Tags::NormalDotFlux<ScalarFieldTag<1>>,
                         ::Tags::NormalDotFlux<ScalarFieldTag<2>>>>
        vars{face_num_points, std::numeric_limits<double>::max()};
    // Inhomogeneous boundary conditions
    elliptic::apply_boundary_condition<false>(
        boundary_condition, box, direction,
        make_not_null(&get<ScalarFieldTag<1>>(vars)),
        make_not_null(&get<ScalarFieldTag<2>>(vars)),
        make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<1>>>(vars)),
        make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<2>>>(vars)));
    // Imposed Dirichlet conditions on field 1
    const auto expected_dirichlet_field = []() -> DataVector {
      if constexpr (Dim == 1) {
        return {1.};
      } else if constexpr (Dim == 2) {
        return {1., 4., 7.};
      } else if constexpr (Dim == 3) {
        return {1., 4., 7., 10., 13., 16., 19., 22., 25.};
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
        return {-20.};
      } else if constexpr (Dim == 2) {
        return {-74., -80., -86.};
      } else if constexpr (Dim == 3) {
        return {-272., -278., -284., -290., -296., -302., -308., -314., -320.};
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
    elliptic::apply_boundary_condition<true>(
        boundary_condition, box, direction,
        make_not_null(&get<ScalarFieldTag<1>>(vars)),
        make_not_null(&get<ScalarFieldTag<2>>(vars)),
        make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<1>>>(vars)),
        make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<2>>>(vars)));
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
