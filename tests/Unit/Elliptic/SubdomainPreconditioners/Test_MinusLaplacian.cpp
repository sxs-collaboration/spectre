// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/SubdomainPreconditioners/MinusLaplacian.hpp"
#include "Elliptic/SubdomainPreconditioners/RegisterDerived.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Options/Options.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarFieldTag {
  using type = Scalar<DataVector>;
};
template <size_t Dim>
struct VectorFieldTag {
  using type = tnsr::I<DataVector, Dim>;
};
template <size_t Dim>
using PoissonSubdomainData =
    LinearSolver::Schwarz::ElementCenteredSubdomainData<
        Dim, tmpl::list<Poisson::Tags::Field>>;
struct OptionsGroup {};

// We don't actually solve the subdomain operator in this test, because it is
// implemented and tested elsewhere. Instead, we test the solver is invoked
// correctly for every tensor component.
template <size_t Dim>
struct TestSolver {
  using options = tmpl::list<>;
  static constexpr Options::String help = "halp";

  template <typename LinearOperator>
  Convergence::HasConverged solve(
      const gsl::not_null<PoissonSubdomainData<Dim>*>
          initial_guess_in_solution_out,
      const LinearOperator& /*linear_operator*/,
      const PoissonSubdomainData<Dim>& source,
      const std::tuple<>& /*operator_args*/ = std::tuple{}) const noexcept {
    // Check the initial guess for each component is sized correctly and zero
    for (size_t i = 0; i < source.element_data.size(); ++i) {
      CHECK(initial_guess_in_solution_out->element_data.data()[i] == 0.);
    }
    for (const auto& [overlap_id, source_data] : source.overlap_data) {
      for (size_t i = 0; i < source_data.size(); ++i) {
        CHECK(initial_guess_in_solution_out->overlap_data.at(overlap_id)
                  .data()[i] == 0.);
      }
    }
    // Keep track of the sources so they can be checked
    sources.push_back(source);
    return {0, 0};
  }
  void reset() noexcept { was_reset = true; }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}

  mutable std::vector<PoissonSubdomainData<Dim>> sources{};
  bool was_reset{false};
};

template <size_t Dim, typename FullData, typename Tag>
void check_component(const PoissonSubdomainData<Dim>& poisson_data,
                     const FullData& expected_data, Tag /*meta*/,
                     const size_t component) {
  CHECK(get(get<Poisson::Tags::Field>(poisson_data.element_data)) ==
        get<std::decay_t<Tag>>(expected_data.element_data)[component]);
  for (const auto& [overlap_id, data] : expected_data.overlap_data) {
    CHECK(get(get<Poisson::Tags::Field>(poisson_data.overlap_data.at(
              overlap_id))) == get<std::decay_t<Tag>>(data)[component]);
  }
}
}  // namespace

namespace elliptic::subdomain_preconditioners {

SPECTRE_TEST_CASE("Unit.Elliptic.SubdomainPreconditioners.MinusLaplacian",
                  "[Unit][Elliptic]") {
  {
    constexpr size_t Dim = 2;
    using LinearSolverType = ::LinearSolver::Serial::LinearSolver<tmpl::list<
        Registrars::MinusLaplacian<Dim, OptionsGroup, TestSolver<Dim>>>>;
    Parallel::register_derived_classes_with_charm<LinearSolverType>();
    const auto created =
        TestHelpers::test_creation<std::unique_ptr<LinearSolverType>>(
            "MinusLaplacian:\n"
            "  Solver:\n");
    REQUIRE(
        dynamic_cast<const MinusLaplacian<Dim, OptionsGroup, TestSolver<Dim>>*>(
            created.get()) != nullptr);
    const auto serialized = serialize_and_deserialize(created);
    const auto cloned = serialized->get_clone();
    const auto& minus_laplacian =
        dynamic_cast<const MinusLaplacian<Dim, OptionsGroup, TestSolver<Dim>>&>(
            *cloned);
    const auto& solver = minus_laplacian.solver();
    // The "real" linear operator is unused, because the solve approximates it
    // with a Laplacian
    const NoSuchType linear_operator{};
    // Manufacture a source with multiple tensors
    using SubdomainData = LinearSolver::Schwarz::ElementCenteredSubdomainData<
        Dim, tmpl::list<ScalarFieldTag, VectorFieldTag<Dim>>>;
    SubdomainData source{};
    source.element_data.initialize(5);
    source
        .overlap_data[std::make_pair(Direction<Dim>::lower_xi(),
                                     ElementId<Dim>{0})]
        .initialize(3);
    std::iota(source.begin(), source.end(), 1.);
    // Apply the solver
    auto initial_guess_in_solution_out =
        make_with_value<SubdomainData>(source, 0.);
    minus_laplacian.solve(make_not_null(&initial_guess_in_solution_out),
                          linear_operator, source);
    // Test the solver was applied to every tensor component in turn
    REQUIRE(solver.sources.size() == 3);
    check_component(solver.sources[0], source, ScalarFieldTag{}, 0);
    check_component(solver.sources[1], source, VectorFieldTag<Dim>{}, 0);
    check_component(solver.sources[2], source, VectorFieldTag<Dim>{}, 1);
  }
  {
    INFO("Factory-create the solver");
    constexpr size_t Dim = 2;
    using LinearSolverType = ::LinearSolver::Serial::LinearSolver<
        tmpl::list<Registrars::MinusLaplacian<Dim, OptionsGroup>>>;
    Parallel::register_derived_classes_with_charm<LinearSolverType>();
    elliptic::subdomain_preconditioners::register_derived_with_charm();
    const auto created =
        TestHelpers::test_creation<std::unique_ptr<LinearSolverType>>(
            "MinusLaplacian:\n"
            "  Solver: ExplicitInverse\n");
    const auto serialized = serialize_and_deserialize(created);
    const auto cloned = serialized->get_clone();
    REQUIRE(dynamic_cast<const MinusLaplacian<Dim, OptionsGroup>*>(
                cloned.get()) != nullptr);
    const auto& minus_laplacian =
        dynamic_cast<const MinusLaplacian<Dim, OptionsGroup>&>(*cloned);
    const auto& solver = minus_laplacian.solver();
    REQUIRE(
        dynamic_cast<
            const LinearSolver::Serial::ExplicitInverse<typename MinusLaplacian<
                Dim, OptionsGroup>::solver_type::registrars>*>(&solver) !=
        nullptr);
  }
  {
    INFO("Resetting");
    MinusLaplacian<1, OptionsGroup, TestSolver<1>> resetting_solver{};
    CHECK_FALSE(resetting_solver.solver().was_reset);
    resetting_solver.reset();
    CHECK(resetting_solver.solver().was_reset);
  }
}

}  // namespace elliptic::subdomain_preconditioners
