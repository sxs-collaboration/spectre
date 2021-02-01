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

template <size_t N>
struct FluxTag : db::SimpleTag {
  using type = tnsr::I<DataVector, 1>;
};

struct System {
  static constexpr size_t volume_dim = 1;
  using primal_fields = tmpl::list<ScalarFieldTag<1>, ScalarFieldTag<2>>;
  using primal_fluxes = tmpl::list<FluxTag<1>, FluxTag<2>>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.BoundaryConditions.AnalyticSolution",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  using Registrar = Registrars::AnalyticSolution<System>;
  const auto boundary_condition_base = TestHelpers::test_factory_creation<
      BoundaryCondition<1, tmpl::list<Registrar>>>(
      "AnalyticSolution:\n"
      "  Field1: Dirichlet\n"
      "  Field2: Neumann");

  // Test semantics
  using BoundaryConditionType =
      typename Registrar::template f<tmpl::list<Registrar>>;
  REQUIRE(dynamic_cast<const BoundaryConditionType*>(
              boundary_condition_base.get()) != nullptr);
  const auto& boundary_condition =
      dynamic_cast<const BoundaryConditionType&>(*boundary_condition_base);
  test_serialization(boundary_condition);
  test_copy_semantics(boundary_condition);
  auto move_boundary_condition = boundary_condition;
  test_move_semantics(std::move(move_boundary_condition), boundary_condition);

  // Test applying the boundary conditions
  const Mesh<1> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  Variables<tmpl::list<
      ::Tags::Analytic<ScalarFieldTag<1>>, ::Tags::Analytic<ScalarFieldTag<2>>,
      ::Tags::Analytic<FluxTag<1>>, ::Tags::Analytic<FluxTag<2>>>>
      analytic_solutions{3};
  std::iota(analytic_solutions.data(),
            analytic_solutions.data() + analytic_solutions.size(), 1.);
  tnsr::i<DataVector, 1> face_normal{1_st};
  get<0>(face_normal) = DataVector{-2.};
  const auto box = db::create<db::AddSimpleTags<
      domain::Tags::Mesh<1>,
      ::Tags::AnalyticSolutions<tmpl::list<ScalarFieldTag<1>, ScalarFieldTag<2>,
                                           FluxTag<1>, FluxTag<2>>>,
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<1>,
                              domain::Tags::Direction<1>>,
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<1>,
          ::Tags::Normalized<
              domain::Tags::UnnormalizedFaceNormal<1, Frame::Inertial>>>>>(
      mesh, std::move(analytic_solutions),
      std::unordered_map<Direction<1>, Direction<1>>{
          {Direction<1>::lower_xi(), Direction<1>::lower_xi()}},
      std::unordered_map<Direction<1>, tnsr::i<DataVector, 1>>{
          {Direction<1>::lower_xi(), face_normal}});
  Variables<tmpl::list<ScalarFieldTag<1>, ScalarFieldTag<2>,
                       ::Tags::NormalDotFlux<ScalarFieldTag<1>>,
                       ::Tags::NormalDotFlux<ScalarFieldTag<2>>>>
      vars{1, std::numeric_limits<double>::max()};
  // Inhomogeneous boundary conditions
  elliptic::apply_boundary_condition<false>(
      boundary_condition, box, Direction<1>::lower_xi(),
      make_not_null(&get<ScalarFieldTag<1>>(vars)),
      make_not_null(&get<ScalarFieldTag<2>>(vars)),
      make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<1>>>(vars)),
      make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<2>>>(vars)));
  // Imposed Dirichlet conditions on field 1
  CHECK_ITERABLE_APPROX(get(get<ScalarFieldTag<1>>(vars)), DataVector{1.});
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::NormalDotFlux<ScalarFieldTag<1>>>(vars)),
      DataVector{std::numeric_limits<double>::max()});
  // Imposed Neumann conditions on field 2
  CHECK_ITERABLE_APPROX(get(get<ScalarFieldTag<2>>(vars)),
                        DataVector{std::numeric_limits<double>::max()});
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::NormalDotFlux<ScalarFieldTag<2>>>(vars)),
      DataVector{-20.});
  // Homogeneous (linearized) boundary conditions
  elliptic::apply_boundary_condition<true>(
      boundary_condition, box, Direction<1>::lower_xi(),
      make_not_null(&get<ScalarFieldTag<1>>(vars)),
      make_not_null(&get<ScalarFieldTag<2>>(vars)),
      make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<1>>>(vars)),
      make_not_null(&get<::Tags::NormalDotFlux<ScalarFieldTag<2>>>(vars)));
  // Imposed Dirichlet conditions on field 1
  CHECK_ITERABLE_APPROX(get(get<ScalarFieldTag<1>>(vars)), DataVector{0.});
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::NormalDotFlux<ScalarFieldTag<1>>>(vars)),
      DataVector{std::numeric_limits<double>::max()});
  // Imposed Neumann conditions on field 2
  CHECK_ITERABLE_APPROX(get(get<ScalarFieldTag<2>>(vars)),
                        DataVector{std::numeric_limits<double>::max()});
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::NormalDotFlux<ScalarFieldTag<2>>>(vars)), DataVector{0.});
}

}  // namespace elliptic::BoundaryConditions
