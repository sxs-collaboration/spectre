// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
// IWYU pragma: no_include <pup.h>
// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tags::div
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {

template <size_t Dim, typename DbTagsList, typename Solution>
void test_first_order_operator(db::DataBox<DbTagsList>&& domain_box,
                               const Solution& solution) {
  using solution_vars_tag =
      Tags::Variables<tmpl::list<Elasticity::Tags::Displacement<Dim>,
                                 Elasticity::Tags::Stress<Dim>>>;
  using operand_vars_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, solution_vars_tag>;
  using operator_vars_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                         operand_vars_tag>;
  using source_vars_tag = db::add_tag_prefix<Tags::Source, solution_vars_tag>;
  using inverse_jacobian_tag =
      Tags::InverseJacobian<Tags::ElementMap<Dim>,
                            Tags::Coordinates<Dim, Frame::Logical>>;
  using deriv_compute_tag =
      Tags::DerivCompute<operand_vars_tag, inverse_jacobian_tag,
                         tmpl::list<LinearSolver::Tags::Operand<
                             Elasticity::Tags::Displacement<Dim>>>>;

  // Retrieve data from the analytic solution
  const auto& inertial_coords =
      get<Tags::Coordinates<Dim, Frame::Inertial>>(domain_box);
  db::item_type<solution_vars_tag> solution_vars{
      get<Tags::Mesh<Dim>>(domain_box).number_of_grid_points()};
  solution_vars.assign_subset(solution.variables(
      inertial_coords, typename solution_vars_tag::tags_list{}));
  db::item_type<operand_vars_tag> operand_vars(solution_vars);
  auto operator_vars = make_with_value<db::item_type<operator_vars_tag>>(
      inertial_coords, std::numeric_limits<double>::signaling_NaN());
  auto source_vars =
      make_with_value<db::item_type<source_vars_tag>>(inertial_coords, 0.);
  source_vars.assign_subset(solution.variables(
      inertial_coords,
      tmpl::list<Tags::Source<Elasticity::Tags::Displacement<Dim>>>{}));

  // Apply operator to the analytic data
  auto argument_box =
      db::create_from<db::RemoveTags<>,
                      db::AddSimpleTags<operand_vars_tag, operator_vars_tag>,
                      db::AddComputeTags<deriv_compute_tag>>(
          std::move(domain_box), std::move(operand_vars),
          std::move(operator_vars));
  db::mutate_apply<
      tmpl::list<
          LinearSolver::Tags::OperatorAppliedTo<
              LinearSolver::Tags::Operand<Elasticity::Tags::Displacement<Dim>>>,
          LinearSolver::Tags::OperatorAppliedTo<
              LinearSolver::Tags::Operand<Elasticity::Tags::Stress<Dim>>>>,
      typename Elasticity::ComputeFirstOrderOperatorAction<Dim>::argument_tags>(
      Elasticity::ComputeFirstOrderOperatorAction<Dim>{},
      make_not_null(&argument_box), solution.constitutive_relation());

  // Test against the source supplied by the analytic solution. These should
  // match on a sufficiently fine mesh, with the difference being the
  // discretization error.
  CHECK_ITERABLE_APPROX(
      get<LinearSolver::Tags::OperatorAppliedTo<
          LinearSolver::Tags::Operand<Elasticity::Tags::Displacement<Dim>>>>(
          argument_box),
      get<Tags::Source<Elasticity::Tags::Displacement<Dim>>>(source_vars));
  CHECK_ITERABLE_APPROX(
      get<LinearSolver::Tags::OperatorAppliedTo<
          LinearSolver::Tags::Operand<Elasticity::Tags::Stress<Dim>>>>(
          argument_box),
      get<Tags::Source<Elasticity::Tags::Stress<Dim>>>(source_vars));
}

template <size_t Dim>
using domain_simple_tags =
    db::AddSimpleTags<Tags::Mesh<Dim>, Tags::ElementMap<Dim>>;

template <size_t Dim>
using domain_compute_tags = db::AddComputeTags<
    Tags::LogicalCoordinates<Dim>,
    Tags::MappedCoordinates<Tags::ElementMap<Dim>,
                            Tags::Coordinates<Dim, Frame::Logical>>,
    Tags::InverseJacobian<Tags::ElementMap<Dim>,
                          Tags::Coordinates<Dim, Frame::Logical>>>;

// Specialize this function for every analytic solution type to set up a
// suitable solution and domain
template <typename SolutionType>
void test_operator_with_analytic_solution();

template <>
void test_operator_with_analytic_solution<Elasticity::Solutions::BentBeam>() {
  const Elasticity::Solutions::BentBeam solution{
      5., 1., 1.,
      // Iron: E=100, nu=0.29
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2>{79.3651,
                                                                 38.7597}};
  // 3 grid points should be enough to represent this solution exactly, since it
  // is a quadratic polynomial
  Mesh<2> mesh_2d{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto};
  ElementMap<2, Frame::Inertial> element_map_2d{
      ElementId<2>{0, make_array<2>(SegmentId{0, 0})},
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Affine>(
              CoordinateMaps::Affine{-1., 1., -solution.length(),
                                     solution.length()},
              CoordinateMaps::Affine{-1., 1., -solution.height(),
                                     solution.height()}))};
  auto domain_box_2d =
      db::create<domain_simple_tags<2>, domain_compute_tags<2>>(
          std::move(mesh_2d), std::move(element_map_2d));

  test_first_order_operator<2>(std::move(domain_box_2d), solution);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Elasticity.FirstOrder",
                  "[Unit][Elliptic]") {
  test_operator_with_analytic_solution<Elasticity::Solutions::BentBeam>();
}
