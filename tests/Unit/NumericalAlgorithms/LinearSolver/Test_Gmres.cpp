// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/LinearSolver/TestHelpers.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
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
    // [gmres_example]
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
    CHECK_FALSE(gmres.has_preconditioner());
    std::vector<double> recorded_residuals;
    const auto has_converged = gmres.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source,
        [&recorded_residuals](
            const Convergence::HasConverged& local_has_converged) {
          recorded_residuals.push_back(
              local_has_converged.residual_magnitude());
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
    // [gmres_example]
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
    const auto check_second_solve = [&linear_operator](
                                        const auto& local_gmres) {
      DenseVector<double> local_initial_guess_in_solution_out{0., 0.};
      const auto local_has_converged = local_gmres.solve(
          make_not_null(&local_initial_guess_in_solution_out), linear_operator,
          DenseVector<double>{2, 1});
      REQUIRE(local_has_converged);
      CHECK(local_has_converged.reason() ==
            Convergence::Reason::AbsoluteResidual);
      CHECK(local_has_converged.num_iterations() == 2);
      const DenseVector<double> expected_local_solution{0.454545454545455,
                                                         0.181818181818182};
      CHECK_ITERABLE_APPROX(local_initial_guess_in_solution_out,
                            expected_local_solution);
    };
    {
      INFO("Check two successive solves with different sources");
      check_second_solve(gmres);
    }
    {
      INFO("Check the solver still works after serialization");
      const auto serialized_gmres = serialize_and_deserialize(gmres);
      check_second_solve(serialized_gmres);
    }
    {
      INFO("Check the solver still works after copying");
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      const auto copied_gmres = gmres;
      check_second_solve(copied_gmres);
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
    const DenseVector<double> expected_solution{0.0909090909090909,
                                                0.6363636363636364};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const auto check_solve = [&linear_operator, &source, &expected_solution](
                                 const auto& local_gmres,
                                 const size_t expected_num_iterations) {
      REQUIRE(local_gmres.has_preconditioner());
      DenseVector<double> local_initial_guess_in_solution_out{2., 1.};
      std::vector<double> local_recorded_residuals;
      const auto local_has_converged = local_gmres.solve(
          make_not_null(&local_initial_guess_in_solution_out), linear_operator,
          source,
          [&local_recorded_residuals](
              const Convergence::HasConverged& recorded_has_converged) {
            local_recorded_residuals.push_back(
                recorded_has_converged.residual_magnitude());
          });
      CAPTURE(local_recorded_residuals);
      REQUIRE(local_has_converged);
      CHECK(local_has_converged.reason() ==
            Convergence::Reason::AbsoluteResidual);
      CHECK(local_has_converged.num_iterations() == expected_num_iterations);
      CHECK_ITERABLE_APPROX(local_initial_guess_in_solution_out,
                            expected_solution);
    };
    {
      INFO("Exact inverse preconditioner");
      // Use the exact inverse of the matrix as preconditioner. This
      // should solve the problem in 1 iteration.
      helpers::ExactInversePreconditioner preconditioner{};
      const Gmres<DenseVector<double>, helpers::ExactInversePreconditioner>
          preconditioned_gmres{convergence_criteria, ::Verbosity::Verbose,
                               std::nullopt, std::move(preconditioner)};
      check_solve(preconditioned_gmres, 1);
      // Check a second solve with the same solver and preconditioner works
      check_solve(preconditioned_gmres, 1);
      {
        INFO("Check that serialization preserves the preconditioner");
        const auto serialized_gmres =
            serialize_and_deserialize(preconditioned_gmres);
        check_solve(serialized_gmres, 1);
      }
      {
        INFO("Check that copying preserves the preconditioner");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const auto copied_gmres = preconditioned_gmres;
        check_solve(copied_gmres, 1);
      }
    }
    {
      INFO("Diagonal (Jacobi) preconditioner");
      // Use the inverse of the diagonal as preconditioner.
      helpers::JacobiPreconditioner preconditioner{};
      const Gmres<DenseVector<double>, helpers::JacobiPreconditioner>
          preconditioned_gmres{convergence_criteria, ::Verbosity::Verbose,
                               std::nullopt, std::move(preconditioner)};
      check_solve(preconditioned_gmres, 2);
    }
    {
      INFO("Richardson preconditioner");
      helpers::RichardsonPreconditioner preconditioner{
          // The optimal relaxation parameter for SPD matrices is 2 / (l_max +
          // l_min) where l_max and l_min are the largest and smallest
          // eigenvalues of the linear operator (see
          // `LinearSolver::Richardson::Richardson`).
          0.2857142857142857,
          // Run two Richardson iterations
          2};
      const Gmres<DenseVector<double>, helpers::RichardsonPreconditioner>
          preconditioned_gmres{convergence_criteria, ::Verbosity::Verbose,
                               std::nullopt, std::move(preconditioner)};
      check_solve(preconditioned_gmres, 1);
    }
    {
      INFO("Nested linear solver as preconditioner");
      // Running another GMRES solver for 2 iterations as preconditioner. It
      // should already solve the problem, so the preconditioned solve only
      // needs a single iteration.
      const Gmres<DenseVector<double>, Gmres<DenseVector<double>>>
          preconditioned_gmres{convergence_criteria,
                               ::Verbosity::Verbose,
                               std::nullopt,
                               {{{2, 0., 0.}, ::Verbosity::Verbose}}};
      check_solve(preconditioned_gmres, 1);
    }
    {
      INFO("Nested factory-created linear solver as preconditioner");
      // Also running another GMRES solver as preconditioner, but passing it as
      // a factory-created abstract `LinearSolver` type.
      using LinearSolverRegistrars =
          tmpl::list<Registrars::Gmres<DenseVector<double>>>;
      using LinearSolverFactory = LinearSolver<LinearSolverRegistrars>;
      const Gmres<DenseVector<double>, LinearSolverFactory,
                  LinearSolverRegistrars>
          preconditioned_gmres{
              convergence_criteria, ::Verbosity::Verbose, std::nullopt,
              std::make_unique<Gmres<DenseVector<double>, LinearSolverFactory,
                                     LinearSolverRegistrars>>(
                  Convergence::Criteria{2, 0., 0.}, ::Verbosity::Verbose)};
      check_solve(preconditioned_gmres, 1);
      {
        INFO("Check that serialization preserves the preconditioner");
        Parallel::register_derived_classes_with_charm<LinearSolverFactory>();
        const auto serialized_gmres =
            serialize_and_deserialize(preconditioned_gmres);
        check_solve(serialized_gmres, 1);
      }
      {
        INFO("Check that copying preserves the preconditioner");
        Parallel::register_derived_classes_with_charm<LinearSolverFactory>();
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const auto copied_gmres = preconditioned_gmres;
        check_solve(copied_gmres, 1);
      }
    }
  }
  {
    INFO("Option-creation");
    {
      const auto solver =
          TestHelpers::test_creation<Gmres<DenseVector<double>>>(
              "ConvergenceCriteria:\n"
              "  MaxIterations: 2\n"
              "  AbsoluteResidual: 0.1\n"
              "  RelativeResidual: 0.5\n"
              "Restart: 50\n"
              "Verbosity: Verbose\n");
      CHECK(solver.convergence_criteria() ==
            Convergence::Criteria{2, 0.1, 0.5});
      CHECK(solver.restart() == 50);
      CHECK(solver.verbosity() == ::Verbosity::Verbose);
      CHECK_FALSE(solver.has_preconditioner());
    }
    {
      const auto solver = TestHelpers::test_creation<
          Gmres<DenseVector<double>, helpers::ExactInversePreconditioner>>(
          "ConvergenceCriteria:\n"
          "  MaxIterations: 2\n"
          "  AbsoluteResidual: 0.1\n"
          "  RelativeResidual: 0.5\n"
          "Restart: None\n"
          "Verbosity: Verbose\n"
          "Preconditioner: None\n");
      CHECK(solver.convergence_criteria() ==
            Convergence::Criteria{2, 0.1, 0.5});
      CHECK(solver.restart() == 2);
      CHECK(solver.verbosity() == ::Verbosity::Verbose);
      CHECK_FALSE(solver.has_preconditioner());
    }
    {
      const auto solver = TestHelpers::test_creation<
          Gmres<DenseVector<double>, helpers::ExactInversePreconditioner>>(
          "ConvergenceCriteria:\n"
          "  MaxIterations: 2\n"
          "  AbsoluteResidual: 0.1\n"
          "  RelativeResidual: 0.5\n"
          "Restart: None\n"
          "Verbosity: Verbose\n"
          "Preconditioner:\n");
      CHECK(solver.convergence_criteria() ==
            Convergence::Criteria{2, 0.1, 0.5});
      CHECK(solver.restart() == 2);
      CHECK(solver.verbosity() == ::Verbosity::Verbose);
      CHECK(solver.has_preconditioner());
    }
    {
      using LinearSolverRegistrars =
          tmpl::list<Registrars::Gmres<DenseVector<double>>>;
      using LinearSolverFactory = LinearSolver<LinearSolverRegistrars>;
      const auto solver =
          TestHelpers::test_factory_creation<LinearSolverFactory>(
              "Gmres:\n"
              "  ConvergenceCriteria:\n"
              "    MaxIterations: 2\n"
              "    AbsoluteResidual: 0.1\n"
              "    RelativeResidual: 0.5\n"
              "  Restart: 50\n"
              "  Verbosity: Verbose\n"
              "  Preconditioner:\n"
              "    Gmres:\n"
              "      ConvergenceCriteria:\n"
              "        MaxIterations: 1\n"
              "        AbsoluteResidual: 0.5\n"
              "        RelativeResidual: 0.9\n"
              "      Restart: None\n"
              "      Verbosity: Verbose\n"
              "      Preconditioner: None\n");
      REQUIRE(solver);
      using Derived = Gmres<DenseVector<double>, LinearSolverFactory,
                            LinearSolverRegistrars>;
      REQUIRE_FALSE(nullptr == dynamic_cast<const Derived*>(solver.get()));
      const auto& derived = dynamic_cast<const Derived&>(*solver);
      CHECK(derived.convergence_criteria() ==
            Convergence::Criteria{2, 0.1, 0.5});
      CHECK(derived.restart() == 50);
      CHECK(derived.verbosity() == ::Verbosity::Verbose);
      REQUIRE(derived.has_preconditioner());
      REQUIRE_FALSE(nullptr ==
                    dynamic_cast<const Derived*>(&derived.preconditioner()));
      const auto& preconditioner =
          dynamic_cast<const Derived&>(derived.preconditioner());
      CHECK(preconditioner.convergence_criteria() ==
            Convergence::Criteria{1, 0.5, 0.9});
      CHECK(preconditioner.restart() == 1);
      CHECK(preconditioner.verbosity() == ::Verbosity::Verbose);
      CHECK_FALSE(preconditioner.has_preconditioner());
      {
        INFO("Copy semantics");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const auto copied_solver = derived;
        CHECK(copied_solver.convergence_criteria() ==
              Convergence::Criteria{2, 0.1, 0.5});
        CHECK(copied_solver.restart() == 50);
        CHECK(copied_solver.verbosity() == ::Verbosity::Verbose);
        REQUIRE(copied_solver.has_preconditioner());
        REQUIRE_FALSE(nullptr == dynamic_cast<const Derived*>(
                                     &copied_solver.preconditioner()));
        const auto& copied_preconditioner =
            dynamic_cast<const Derived&>(copied_solver.preconditioner());
        CHECK(copied_preconditioner.convergence_criteria() ==
              Convergence::Criteria{1, 0.5, 0.9});
        CHECK(copied_preconditioner.restart() == 1);
        CHECK(copied_preconditioner.verbosity() == ::Verbosity::Verbose);
        CHECK_FALSE(copied_preconditioner.has_preconditioner());
      }
    }
  }
}

}  // namespace LinearSolver::Serial
