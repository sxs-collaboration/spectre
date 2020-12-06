// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/LinearSolver/TestHelpers.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "Utilities/Gsl.hpp"

namespace helpers = TestHelpers::LinearSolver;

namespace LinearSolver::Serial {

namespace {
struct ScalarField : db::SimpleTag {
  using type = Scalar<DataVector>;
};
template <typename Tag>
struct SomePrefix : db::PrefixTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.LinearSolver.Serial.Gmres",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  {
    /// [gmres_example]
    INFO("Solve a symmetric 2x2 matrix");
    DenseMatrix<double> matrix(2, 2);
    matrix(0, 0) = 4.;
    matrix(0, 1) = 1.;
    matrix(1, 0) = 1.;
    matrix(1, 1) = 3.;
    const helpers::ApplyMatrix linear_operator{std::move(matrix)};
    const DenseVector<double> source{1., 2.};
    DenseVector<double> initial_guess_in_solution_out{2., 1.};
    const DenseVector<double> expected_solution{0.0909090909090909,
                                                0.6363636363636364};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<DenseVector<double>> gmres{convergence_criteria,
                                           ::Verbosity::Verbose};
    std::vector<double> recorded_residuals;
    const auto has_converged = gmres.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source,
        NoPreconditioner{},
        [&recorded_residuals](const Convergence::HasConverged& has_converged) {
          recorded_residuals.push_back(has_converged.residual_magnitude());
        });
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK(has_converged.residual_magnitude() <= 1.e-14);
    CHECK(has_converged.initial_residual_magnitude() ==
          approx(8.54400374531753));
    CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
    // The residuals should decrease monotonically
    CHECK(recorded_residuals[0] == has_converged.initial_residual_magnitude());
    for (size_t i = 1; i < has_converged.num_iterations(); ++i) {
      CHECK(recorded_residuals[i] <= recorded_residuals[i - 1]);
    }
    /// [gmres_example]
    {
      INFO("Check that a solved system terminates early");
      const auto second_has_converged =
          gmres.solve(make_not_null(&initial_guess_in_solution_out),
                      linear_operator, source);
      REQUIRE(second_has_converged);
      CHECK(second_has_converged.reason() ==
            Convergence::Reason::AbsoluteResidual);
      CHECK(second_has_converged.num_iterations() == 0);
      CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
    }
    {
      INFO("Check two successive solves with different sources");
      DenseVector<double> second_initial_guess_in_solution_out{0., 0.};
      const auto second_has_converged =
          gmres.solve(make_not_null(&second_initial_guess_in_solution_out),
                      linear_operator, DenseVector<double>{2, 1});
      REQUIRE(second_has_converged);
      CHECK(second_has_converged.reason() ==
            Convergence::Reason::AbsoluteResidual);
      CHECK(second_has_converged.num_iterations() == 2);
      const DenseVector<double> expected_second_solution{0.454545454545455,
                                                         0.181818181818182};
      CHECK_ITERABLE_APPROX(second_initial_guess_in_solution_out,
                            expected_second_solution);
    }
  }
  {
    INFO("Solve a non-symmetric 2x2 matrix");
    DenseMatrix<double> matrix(2, 2);
    matrix(0, 0) = 4.;
    matrix(0, 1) = 1.;
    matrix(1, 0) = 3.;
    matrix(1, 1) = 1.;
    const helpers::ApplyMatrix linear_operator{std::move(matrix)};
    const DenseVector<double> source{1., 2.};
    DenseVector<double> initial_guess_in_solution_out{2., 1.};
    const DenseVector<double> expected_solution{-1., 5.};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<DenseVector<double>> gmres{convergence_criteria,
                                           ::Verbosity::Verbose};
    const auto has_converged = gmres.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
  }
  {
    INFO("Solve a matrix-free linear operator with Variables");
    using Vars = Variables<tmpl::list<ScalarField>>;
    constexpr size_t num_points = 2;
    // This also tests that the linear operator can be a lambda
    const auto linear_operator = [](const gsl::not_null<Vars*> result,
                                    const Vars& operand) noexcept {
      if (result->number_of_grid_points() != num_points) {
        result->initialize(num_points);
      }
      const auto& data = get(get<ScalarField>(operand));
      get(get<ScalarField>(*result)) =
          DataVector{data[0] * 4. + data[1], data[0] * 3. + data[1]};
    };
    // Adding a prefix to make sure prefixed sources work as well
    Variables<tmpl::list<SomePrefix<ScalarField>>> source{num_points};
    get(get<SomePrefix<ScalarField>>(source)) = DataVector{1., 2.};
    Vars initial_guess_in_solution_out{num_points};
    get(get<ScalarField>(initial_guess_in_solution_out)) = DataVector{2., 1.};
    const DataVector expected_solution{-1., 5.};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<Vars> gmres{convergence_criteria, ::Verbosity::Verbose};
    const auto has_converged = gmres.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK_ITERABLE_APPROX(get(get<ScalarField>(initial_guess_in_solution_out)),
                          expected_solution);
  }
  {
    INFO("Restarting");
    DenseMatrix<double> matrix(3, 3);
    matrix(0, 0) = 4.;
    matrix(0, 1) = 1.;
    matrix(0, 2) = 1.;
    matrix(1, 0) = 1.;
    matrix(1, 1) = 1.;
    matrix(1, 2) = 3.;
    matrix(2, 0) = 0.;
    matrix(2, 1) = 2.;
    matrix(2, 2) = 0.;
    const helpers::ApplyMatrix linear_operator{std::move(matrix)};
    const DenseVector<double> source{1., 2., 1.};
    DenseVector<double> initial_guess_in_solution_out{2., 1., 0.};
    const DenseVector<double> expected_solution{0., 0.5, 0.5};
    const Convergence::Criteria convergence_criteria{100, 1.e-14, 0.};
    // Restart every other iteration. The algorithm would converge in 3
    // iterations without restarting, so restarting is of course ridiculously
    // inefficient for this problem size. The number of iterations rises to 59.
    const size_t restart = 2;
    const Gmres<DenseVector<double>> gmres{convergence_criteria,
                                           ::Verbosity::Verbose, restart};
    const auto has_converged = gmres.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 59);
    CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
  }
  {
    INFO("Preconditioning");
    DenseMatrix<double> matrix(2, 2);
    matrix(0, 0) = 4.;
    matrix(0, 1) = 1.;
    matrix(1, 0) = 1.;
    matrix(1, 1) = 3.;
    const helpers::ApplyMatrix linear_operator{std::move(matrix)};
    const DenseVector<double> source{1., 2.};
    DenseVector<double> initial_guess_in_solution_out{2., 1.};
    const DenseVector<double> expected_solution{0.0909090909090909,
                                                0.6363636363636364};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<DenseVector<double>> gmres{convergence_criteria,
                                           ::Verbosity::Verbose};
    SECTION("Exact inverse preconditioner") {
      // Use the exact inverse of the matrix as preconditioner. This
      // should solve the problem in 1 iteration.
      const helpers::ExactInversePreconditioner preconditioner{};
      const auto has_converged =
          gmres.solve(make_not_null(&initial_guess_in_solution_out),
                      linear_operator, source, preconditioner);
      REQUIRE(has_converged);
      CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
      CHECK(has_converged.num_iterations() == 1);
      CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
    }
    SECTION("Diagonal (Jacobi) preconditioner") {
      // Use the inverse of the diagonal as preconditioner.
      const helpers::JacobiPreconditioner preconditioner{};
      const auto has_converged =
          gmres.solve(make_not_null(&initial_guess_in_solution_out),
                      linear_operator, source, preconditioner);
      REQUIRE(has_converged);
      CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
      CHECK(has_converged.num_iterations() == 2);
      CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
    }
    SECTION("Richardson preconditioner") {
      const helpers::RichardsonPreconditioner preconditioner{
          // The optimal relaxation parameter for SPD matrices is 2 / (l_max +
          // l_min) where l_max and l_min are the largest and smallest
          // eigenvalues of the linear operator (see
          // `LinearSolver::Richardson::Richardson`).
          0.2857142857142857,
          // Run two Richardson iterations
          2};
      std::vector<double> recorded_residuals;
      const auto has_converged = gmres.solve(
          make_not_null(&initial_guess_in_solution_out), linear_operator,
          source, preconditioner,
          [&recorded_residuals](
              const Convergence::HasConverged& has_converged) {
            recorded_residuals.push_back(has_converged.residual_magnitude());
          });
      CAPTURE(recorded_residuals);
      REQUIRE(has_converged);
      CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
      CHECK(has_converged.num_iterations() == 1);
      CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
    }
  }
}

}  // namespace LinearSolver::Serial
