// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "Utilities/Gsl.hpp"

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
    const DenseVector<double> source{1., 2.};
    const DenseVector<double> initial_guess{2., 1.};
    const DenseVector<double> expected_solution{0.0909090909090909,
                                                0.6363636363636364};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<DenseVector<double>> gmres{convergence_criteria,
                                           ::Verbosity::Verbose};
    const auto linear_operator =
        [&matrix](const DenseVector<double>& arg) noexcept {
          return matrix * arg;
        };
    std::vector<double> recorded_residuals;
    const auto result = gmres(
        linear_operator, source, initial_guess,
        IdentityPreconditioner<DenseVector<double>>{},
        [&recorded_residuals](const Convergence::HasConverged& has_converged) {
          recorded_residuals.push_back(has_converged.residual_magnitude());
        });
    const auto& has_converged = result.first;
    const auto& solution = result.second;
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK(has_converged.residual_magnitude() <= 1.e-14);
    CHECK(has_converged.initial_residual_magnitude() ==
          approx(8.54400374531753));
    CHECK_ITERABLE_APPROX(solution, expected_solution);
    // The residuals should decrease monotonically
    CHECK(recorded_residuals[0] == has_converged.initial_residual_magnitude());
    for (size_t i = 1; i < has_converged.num_iterations(); ++i) {
      CHECK(recorded_residuals[i] <= recorded_residuals[i - 1]);
    }
    /// [gmres_example]
    {
      INFO("Check that a solved system terminates early");
      const auto second_result = gmres(linear_operator, source, solution);
      const auto& second_has_converged = second_result.first;
      const auto& second_solution = second_result.second;
      REQUIRE(second_has_converged);
      CHECK(second_has_converged.reason() ==
            Convergence::Reason::AbsoluteResidual);
      CHECK(second_has_converged.num_iterations() == 0);
      CHECK_ITERABLE_APPROX(solution, second_solution);
    }
    {
      INFO("Check two successive solves with different sources");
      const auto second_result =
          gmres(linear_operator, DenseVector<double>{2, 1},
                DenseVector<double>{0., 0.});
      const auto& second_has_converged = second_result.first;
      const auto& second_solution = second_result.second;
      REQUIRE(second_has_converged);
      CHECK(second_has_converged.reason() ==
            Convergence::Reason::AbsoluteResidual);
      CHECK(second_has_converged.num_iterations() == 2);
      const DenseVector<double> expected_second_solution{0.454545454545455,
                                                         0.181818181818182};
      CHECK_ITERABLE_APPROX(second_solution, expected_second_solution);
    }
  }
  {
    INFO("Solve a non-symmetric 2x2 matrix");
    DenseMatrix<double> matrix(2, 2);
    matrix(0, 0) = 4.;
    matrix(0, 1) = 1.;
    matrix(1, 0) = 3.;
    matrix(1, 1) = 1.;
    const DenseVector<double> source{1., 2.};
    const DenseVector<double> initial_guess{2., 1.};
    const DenseVector<double> expected_solution{-1., 5.};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<DenseVector<double>> gmres{convergence_criteria,
                                           ::Verbosity::Verbose};
    const auto result = gmres(
        [&matrix](const DenseVector<double>& arg) noexcept {
          return matrix * arg;
        },
        source, initial_guess);
    const auto& has_converged = result.first;
    const auto& solution = result.second;
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK_ITERABLE_APPROX(solution, expected_solution);
  }
  {
    INFO("Solve a matrix-free linear operator with Variables");
    using Vars = Variables<tmpl::list<ScalarField>>;
    constexpr size_t num_points = 2;
    const auto linear_operator = [](const Vars& arg) noexcept {
      const auto& data = get(get<ScalarField>(arg));
      Vars result{num_points};
      get(get<ScalarField>(result)) =
          DataVector{data[0] * 4. + data[1], data[0] * 3. + data[1]};
      return result;
    };
    // Adding a prefix to make sure prefixed sources work as well
    Variables<tmpl::list<SomePrefix<ScalarField>>> source{num_points};
    get(get<SomePrefix<ScalarField>>(source)) = DataVector{1., 2.};
    Vars initial_guess{num_points};
    get(get<ScalarField>(initial_guess)) = DataVector{2., 1.};
    const DataVector expected_solution{-1., 5.};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<Vars> gmres{convergence_criteria, ::Verbosity::Verbose};
    const auto result = gmres(linear_operator, source, initial_guess);
    const auto& has_converged = result.first;
    const auto& solution = result.second;
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK_ITERABLE_APPROX(get(get<ScalarField>(solution)), expected_solution);
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
    const DenseVector<double> source{1., 2., 1.};
    const DenseVector<double> initial_guess{2., 1., 0.};
    const DenseVector<double> expected_solution{0., 0.5, 0.5};
    const Convergence::Criteria convergence_criteria{100, 1.e-14, 0.};
    // Restart every other iteration. The algorithm would converge in 3
    // iterations without restarting, so restarting is of course ridiculously
    // inefficient for this problem size. The number of iterations rises to 59.
    const size_t restart = 2;
    const Gmres<DenseVector<double>> gmres{convergence_criteria,
                                           ::Verbosity::Verbose, restart};
    const auto result = gmres(
        [&matrix](const DenseVector<double>& arg) noexcept {
          return matrix * arg;
        },
        source, initial_guess);
    const auto& has_converged = result.first;
    const auto& solution = result.second;
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 59);
    CHECK_ITERABLE_APPROX(solution, expected_solution);
  }
  {
    INFO("Preconditioning");
    DenseMatrix<double> matrix(2, 2);
    matrix(0, 0) = 4.;
    matrix(0, 1) = 1.;
    matrix(1, 0) = 1.;
    matrix(1, 1) = 3.;
    const auto linear_operator =
        [&matrix](const DenseVector<double>& arg) noexcept {
          return matrix * arg;
        };
    const DenseVector<double> source{1., 2.};
    const DenseVector<double> initial_guess{2., 1.};
    const DenseVector<double> expected_solution{0.0909090909090909,
                                                0.6363636363636364};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<DenseVector<double>> gmres{convergence_criteria,
                                           ::Verbosity::Verbose};
    {
      INFO("Exact inverse preconditioner");
      const DenseMatrix<double> matrix_inv = blaze::inv(matrix);
      const auto result =
          gmres(linear_operator, source, initial_guess,
                // Use the exact inverse of the matrix as preconditioner. This
                // should solve the problem in 1 iteration.
                [&matrix_inv](const DenseVector<double>& arg) noexcept {
                  return matrix_inv * arg;
                });
      const auto& has_converged = result.first;
      const auto& solution = result.second;
      REQUIRE(has_converged);
      CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
      CHECK(has_converged.num_iterations() == 1);
      CHECK_ITERABLE_APPROX(solution, expected_solution);
    }
    {
      INFO("Diagonal (Jacobi) preconditioner");
      // Use the inverse of the diagonal as preconditioner.
      const auto preconditioner = [](DenseVector<double> arg) noexcept {
        arg[0] *= 0.25;
        arg[1] *= 1. / 3.;
        return arg;
      };
      const auto result =
          gmres(linear_operator, source, initial_guess, preconditioner);
      const auto& has_converged = result.first;
      const auto& solution = result.second;
      REQUIRE(has_converged);
      CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
      CHECK(has_converged.num_iterations() == 2);
      CHECK_ITERABLE_APPROX(solution, expected_solution);
    }
    {
      INFO("Richardson preconditioner");
      std::vector<double> recorded_residuals;
      // Run a few Richardson iterations as preconditioner.
      // The relaxation parameter is 2 / (l_max + l_min) where l_max and l_min
      // are the largest and smallest eigenvalues of the linear operator
      // (see `LinearSolver::Richardson::Richardson`).
      const double relaxation_parameter = 0.2857142857142857;
      const auto preconditioner =
          [&linear_operator, &relaxation_parameter](
              const DenseVector<double>& local_source) noexcept {
            DenseVector<double> result(local_source.size(), 0.);
            for (size_t i = 0; i < 2; ++i) {
              result += relaxation_parameter *
                        (local_source - linear_operator(result));
            }
            return result;
          };
      const auto result = gmres(
          linear_operator, source, initial_guess, preconditioner,
          [&recorded_residuals](
              const Convergence::HasConverged& has_converged) {
            recorded_residuals.push_back(has_converged.residual_magnitude());
          });
      const auto& has_converged = result.first;
      const auto& solution = result.second;
      CAPTURE(recorded_residuals);
      REQUIRE(has_converged);
      CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
      CHECK(has_converged.num_iterations() == 1);
      CHECK_ITERABLE_APPROX(solution, expected_solution);
    }
  }
}

}  // namespace LinearSolver::Serial
