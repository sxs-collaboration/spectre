// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/ScalarWave/Constraints.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {
// Test the return-by-value one-index constraint function using random values
template <size_t SpatialDim>
void test_one_index_constraint_random(
    const DataVector& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::i<DataVector, SpatialDim, Frame::Inertial> (*)(
          const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&,
          const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&)>(
          &ScalarWave::one_index_constraint<SpatialDim>),
      "Constraints", "one_index_constraint", {{{-10.0, 10.0}}}, used_for_size);
  TestHelpers::db::test_compute_tag<
      ScalarWave::Tags::OneIndexConstraintCompute<SpatialDim>>(
      "OneIndexConstraint");
}

// Test the return-by-value two-index constraint function using random values
template <size_t SpatialDim>
void test_two_index_constraint_random(
    const DataVector& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ij<DataVector, SpatialDim, Frame::Inertial> (*)(
          const tnsr::ij<DataVector, SpatialDim, Frame::Inertial>&)>(
          &ScalarWave::two_index_constraint<SpatialDim>),
      "Constraints", "two_index_constraint", {{{-10.0, 10.0}}}, used_for_size,
      1.0e-12);
  TestHelpers::db::test_compute_tag<
      ScalarWave::Tags::TwoIndexConstraintCompute<SpatialDim>>(
      "TwoIndexConstraint");
}

template <typename Solution>
void test_constraints_and_compute_tags_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound,
    const double error_tolerance) noexcept {
  constexpr size_t spatial_dim = 3;

  // Check vs. time-independent analytic solution
  // Set up grid
  Mesh<3> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  const size_t data_size = mesh.number_of_grid_points();

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1.0, 1.0, lower_bound[0], upper_bound[0]},
          Affine{-1.0, 1.0, lower_bound[1], upper_bound[1]},
          Affine{-1.0, 1.0, lower_bound[2], upper_bound[2]},
      });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = 0.;

  // Evaluate analytic solution
  const auto vars = solution.variables(
      x, t,
      tmpl::list<ScalarWave::Pi, ScalarWave::Phi<spatial_dim>,
                 ScalarWave::Psi>{});
  const auto& psi = get<ScalarWave::Psi>(vars);
  const auto& phi = get<ScalarWave::Phi<spatial_dim>>(vars);
  const auto& pi = get<ScalarWave::Pi>(vars);
  // Compute derivatives d_phi and d_psi numerically
  // First, prepare
  using variables_tags_to_differentiate =
      tmpl::list<ScalarWave::Psi, ScalarWave::Phi<3>>;
  Variables<variables_tags_to_differentiate> local_vars(data_size);
  get<ScalarWave::Psi>(local_vars) = psi;
  get<ScalarWave::Phi<spatial_dim>>(local_vars) = phi;
  // Second, compute derivatives
  const auto local_derivs =
      partial_derivatives<variables_tags_to_differentiate>(
          local_vars, mesh, coord_map.inv_jacobian(x_logical));
  const auto& deriv_psi = get<
      Tags::deriv<ScalarWave::Psi, tmpl::size_t<spatial_dim>, Frame::Inertial>>(
      local_derivs);
  const auto& deriv_phi =
      get<Tags::deriv<ScalarWave::Phi<spatial_dim>, tmpl::size_t<spatial_dim>,
                      Frame::Inertial>>(local_derivs);

  // (1.) Get the constraints, and check that they vanish to error_tolerance
  auto one_index_constraint =
      make_with_value<tnsr::i<DataVector, spatial_dim, Frame::Inertial>>(
          x, std::numeric_limits<double>::signaling_NaN());
  ScalarWave::one_index_constraint(make_not_null(&one_index_constraint),
                                   deriv_psi, phi);
  auto two_index_constraint =
      make_with_value<tnsr::ij<DataVector, spatial_dim, Frame::Inertial>>(
          x, std::numeric_limits<double>::signaling_NaN());
  ScalarWave::two_index_constraint(make_not_null(&two_index_constraint),
                                   deriv_phi);

  Approx numerical_approx = Approx::custom().epsilon(error_tolerance).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(
      one_index_constraint,
      (make_with_value<tnsr::i<DataVector, spatial_dim, Frame::Inertial>>(x,
                                                                          0.)),
      numerical_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      two_index_constraint,
      (make_with_value<tnsr::ij<DataVector, spatial_dim, Frame::Inertial>>(x,
                                                                           0.)),
      numerical_approx);

  // (2.) Test that compute tags return expected values
  const auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Coordinates<spatial_dim, Frame::Inertial>,
          ScalarWave::Psi, ScalarWave::Phi<spatial_dim>, ScalarWave::Pi,
          ::Tags::deriv<ScalarWave::Psi, tmpl::size_t<spatial_dim>,
                        Frame::Inertial>,
          ::Tags::deriv<ScalarWave::Phi<spatial_dim>, tmpl::size_t<spatial_dim>,
                        Frame::Inertial>>,
      db::AddComputeTags<
          ScalarWave::Tags::OneIndexConstraintCompute<spatial_dim>,
          ScalarWave::Tags::TwoIndexConstraintCompute<spatial_dim>>>(
      x, psi, phi, pi, deriv_psi, deriv_phi);

  // Check that their compute items in databox furnish identical values
  CHECK(db::get<ScalarWave::Tags::OneIndexConstraint<spatial_dim>>(box) ==
        one_index_constraint);
  CHECK(db::get<ScalarWave::Tags::TwoIndexConstraint<spatial_dim>>(box) ==
        two_index_constraint);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.Constraints",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarWave/"};
  {
    INFO("Testing constraints with randomized input");
    // Test the one-index constraint with random numbers
    GENERATE_UNINITIALIZED_DATAVECTOR;
    CHECK_FOR_DATAVECTORS(test_one_index_constraint_random, (1, 2, 3));
    CHECK_FOR_DATAVECTORS(test_two_index_constraint_random, (1, 2, 3));
  }
  {
    INFO("Testing constraints for a spherical-wave analytic solution");
    const size_t grid_size = 8;
    const std::array<double, 3> upper_bound{{6., 6., 6.}};
    const std::array<double, 3> lower_bound{{0., 0., 0.}};

    const ScalarWave::Solutions::RegularSphericalWave solution(
        std::make_unique<MathFunctions::Gaussian>(1., 1., 0.));

    // Note: looser numerical tolerance because this check
    // uses numerical derivatives
    test_constraints_and_compute_tags_analytic(
        solution, grid_size, lower_bound, upper_bound,
        std::numeric_limits<double>::epsilon() * 1.e6);
  }
}
