// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/SubdomainPreconditioners/MinusLaplacian.hpp"
#include "Elliptic/SubdomainPreconditioners/RegisterDerived.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};
template <size_t Dim>
struct VectorFieldTag : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};
template <size_t Dim>
using PoissonSubdomainData =
    LinearSolver::Schwarz::ElementCenteredSubdomainData<
        Dim, tmpl::list<Poisson::Tags::Field>>;
struct OptionsGroup {};

template <size_t Dim>
struct BoundaryCondition
    : elliptic::BoundaryConditions::BoundaryCondition<Dim> {
  explicit BoundaryCondition(
      std::vector<elliptic::BoundaryConditionType> bc_types)
      : bc_types_(std::move(bc_types)) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BoundaryCondition);
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<BoundaryCondition>(*this);
  }
  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return bc_types_;
  }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    elliptic::BoundaryConditions::BoundaryCondition<Dim>::pup(p);
    p | bc_types_;
  }

 private:
  std::vector<elliptic::BoundaryConditionType> bc_types_;
};

template <size_t Dim>
PUP::able::PUP_ID BoundaryCondition<Dim>::my_PUP_ID = 0;  // NOLINT

// We don't actually solve the subdomain operator in this test, because it is
// implemented and tested elsewhere. Instead, we test the solver is invoked
// correctly for every tensor component.
template <size_t Dim>
struct TestSolver {
  using options = tmpl::list<>;
  static constexpr Options::String help = "halp";

  template <typename LinearOperator, typename... OperatorArgs>
  Convergence::HasConverged solve(
      const gsl::not_null<PoissonSubdomainData<Dim>*>
          initial_guess_in_solution_out,
      const LinearOperator& /*linear_operator*/,
      const PoissonSubdomainData<Dim>& source,
      const std::tuple<OperatorArgs...>& operator_args) const {
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
    // Check the boundary conditions
    const auto& boundary_conditions = get<1>(operator_args);
    std::map<std::pair<size_t, Direction<Dim>>, elliptic::BoundaryConditionType>
        local_bc_types{};
    for (const auto& [boundary_id, bc] : boundary_conditions) {
      const auto robin_bc =
          dynamic_cast<const Poisson::BoundaryConditions::Robin<Dim>*>(&bc);
      REQUIRE(robin_bc != nullptr);
      CHECK(robin_bc->constant() == 0.);
      CHECK(((robin_bc->dirichlet_weight() == 1. and
              robin_bc->neumann_weight() == 0.) or
             (robin_bc->dirichlet_weight() == 0. and
              robin_bc->neumann_weight() == 1.)));
      local_bc_types.emplace(boundary_id,
                             robin_bc->neumann_weight() == 1.
                                 ? elliptic::BoundaryConditionType::Neumann
                                 : elliptic::BoundaryConditionType::Dirichlet);
    }
    bc_types.push_back(std::move(local_bc_types));
    // Keep track of the sources so they can be checked
    sources.push_back(source);
    return {0, 0};
  }
  void reset() {
    sources.clear();
    bc_types.clear();
    was_reset = true;
  }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::vector<PoissonSubdomainData<Dim>> sources{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::vector<std::map<std::pair<size_t, Direction<Dim>>,
                               elliptic::BoundaryConditionType>>
      bc_types{};
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

template <size_t Dim>
auto make_block_map() {
  return domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
      domain::CoordinateMaps::Identity<Dim>{});
}

auto make_databox_with_boundary_conditions() {
  constexpr size_t Dim = 2;
  // Subdomain geometry:
  //
  //             D / D / D
  //              vvvvv
  //             +---+-+
  // N / D / D > |   | |    (Block 0)
  //             +---+-+
  // D / N / N > |   |      (Block 1)
  //             +---+
  //
  // - Two blocks, three tensor components (a scalar and a vector). Block 0 has
  //   an external boundary at the top with Dirichlet conditions for all
  //   components, and an external boundary to the left with Neumann conditions
  //   for the scalar and Dirichlet conditions for the vector. Block 1 has an
  //   external boundary to the left with the reverse.
  // - The subdomain is centered on the top-left element of the domain.
  //   Horizontally, it overlaps with another element of Block 0 toward the
  //   right. Vertically, the element spans the entire Block 0 and the subdomain
  //   overlaps with an element of Block 1. The domain extends further toward
  //   the right and the bottom.
  // - We have two distinct boundary condition signatures: (D, N, D) for the
  //   the scalar, and (D, D, N) for both vector components, where the three
  //   entries in the signature refer to the three external boundaries in the
  //   order (Block 0 top, Block 0 left, Block 1 left).

  // Boundary conditions
  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{2};
  boundary_conditions[0][Direction<Dim>::upper_eta()] =
      std::make_unique<BoundaryCondition<Dim>>(BoundaryCondition<Dim>{
          {3, elliptic::BoundaryConditionType::Dirichlet}});
  boundary_conditions[0][Direction<Dim>::lower_xi()] =
      std::make_unique<BoundaryCondition<Dim>>(
          BoundaryCondition<Dim>{{elliptic::BoundaryConditionType::Neumann,
                                  elliptic::BoundaryConditionType::Dirichlet,
                                  elliptic::BoundaryConditionType::Dirichlet}});
  boundary_conditions[1][Direction<Dim>::lower_xi()] =
      std::make_unique<BoundaryCondition<Dim>>(
          BoundaryCondition<Dim>{{elliptic::BoundaryConditionType::Dirichlet,
                                  elliptic::BoundaryConditionType::Neumann,
                                  elliptic::BoundaryConditionType::Neumann}});
  // only needed for completeness, not used in test
  boundary_conditions[0][Direction<Dim>::upper_xi()] =
      std::make_unique<BoundaryCondition<Dim>>(BoundaryCondition<Dim>{
          {3, elliptic::BoundaryConditionType::Dirichlet}});
  boundary_conditions[1][Direction<Dim>::upper_xi()] =
      std::make_unique<BoundaryCondition<Dim>>(BoundaryCondition<Dim>{
          {3, elliptic::BoundaryConditionType::Dirichlet}});
  boundary_conditions[1][Direction<Dim>::lower_eta()] =
      std::make_unique<BoundaryCondition<Dim>>(BoundaryCondition<Dim>{
          {3, elliptic::BoundaryConditionType::Dirichlet}});
  // Blocks
  Block<Dim> block_0{
      make_block_map<Dim>(), 0, {{Direction<Dim>::lower_eta(), {1, {}}}}};
  Block<Dim> block_1{
      make_block_map<Dim>(), 1, {{Direction<Dim>::upper_eta(), {0, {}}}}};
  // Domain
  std::vector<Block<Dim>> blocks{};
  blocks.emplace_back(std::move(block_0));
  blocks.emplace_back(std::move(block_1));
  Domain<Dim> domain{std::move(blocks)};
  // Refinement
  const std::vector<std::array<size_t, Dim>> refinement{{{2, 0}}, {{1, 1}}};
  // Elements
  const ElementId<Dim> central_element_id{0, {{{2, 0}, {0, 0}}}};
  const ElementId<Dim> right_element_id{0, {{{2, 1}, {0, 0}}}};
  const ElementId<Dim> bottom_element_id{1, {{{1, 0}, {1, 1}}}};
  Element<Dim> central_element = domain::Initialization::create_initial_element(
      central_element_id, domain.blocks()[0], refinement);
  Element<Dim> right_element = domain::Initialization::create_initial_element(
      right_element_id, domain.blocks()[0], refinement);
  Element<Dim> bottom_element = domain::Initialization::create_initial_element(
      bottom_element_id, domain.blocks()[1], refinement);
  // Subdomain
  LinearSolver::Schwarz::OverlapMap<Dim, Element<Dim>> overlap_elements{
      {{Direction<Dim>::upper_xi(), right_element_id},
       std::move(right_element)},
      {{Direction<Dim>::lower_eta(), bottom_element_id},
       std::move(bottom_element)}};
  return db::create<tmpl::list<
      domain::Tags::ExternalBoundaryConditions<Dim>, domain::Tags::Element<Dim>,
      LinearSolver::Schwarz::Tags::Overlaps<domain::Tags::Element<Dim>, Dim,
                                            OptionsGroup>>>(
      std::move(boundary_conditions), std::move(central_element),
      std::move(overlap_elements));
}

auto make_databox_without_boundary_conditions() {
  constexpr size_t Dim = 2;
  Block<Dim> block{make_block_map<Dim>(), 0, {}};
  std::vector<Block<Dim>> blocks{};
  blocks.emplace_back(std::move(block));
  Domain<Dim> domain{std::move(blocks)};
  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{domain.blocks().size()};
  const std::vector<std::array<size_t, Dim>> refinement{{{2, 2}}};
  const ElementId<Dim> central_element_id{0, {{{2, 1}, {2, 1}}}};
  const ElementId<Dim> right_element_id{0, {{{2, 2}, {2, 1}}}};
  Element<Dim> central_element = domain::Initialization::create_initial_element(
      central_element_id, domain.blocks()[0], refinement);
  Element<Dim> right_element = domain::Initialization::create_initial_element(
      right_element_id, domain.blocks()[0], refinement);
  LinearSolver::Schwarz::OverlapMap<Dim, Element<Dim>> overlap_elements{
      {{Direction<Dim>::upper_xi(), right_element_id},
       std::move(right_element)}};
  return db::create<tmpl::list<
      domain::Tags::ExternalBoundaryConditions<Dim>, domain::Tags::Element<Dim>,
      LinearSolver::Schwarz::Tags::Overlaps<domain::Tags::Element<Dim>, Dim,
                                            OptionsGroup>>>(
      std::move(boundary_conditions), std::move(central_element),
      std::move(overlap_elements));
}
}  // namespace

namespace elliptic::subdomain_preconditioners {

SPECTRE_TEST_CASE("Unit.Elliptic.SubdomainPreconditioners.MinusLaplacian",
                  "[Unit][Elliptic]") {
  {
    constexpr size_t Dim = 2;
    using LinearSolverType = ::LinearSolver::Serial::LinearSolver<tmpl::list<
        Registrars::MinusLaplacian<Dim, OptionsGroup, TestSolver<Dim>>>>;
    register_derived_classes_with_charm<LinearSolverType>();
    const auto created =
        TestHelpers::test_creation<std::unique_ptr<LinearSolverType>>(
            "MinusLaplacian:\n"
            "  Solver:\n"
            "  BoundaryConditions: Auto\n");
    REQUIRE(
        dynamic_cast<const MinusLaplacian<Dim, OptionsGroup, TestSolver<Dim>>*>(
            created.get()) != nullptr);
    const auto serialized = serialize_and_deserialize(created);
    auto cloned = serialized->get_clone();
    auto& minus_laplacian =
        dynamic_cast<MinusLaplacian<Dim, OptionsGroup, TestSolver<Dim>>&>(
            *cloned);
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
    using BcSignature = typename std::decay_t<
        decltype(minus_laplacian)>::BoundaryConditionsSignature;
    {
      INFO("Subdomain with no external boundaries");
      minus_laplacian.solve(
          make_not_null(&initial_guess_in_solution_out), linear_operator,
          source, std::make_tuple(make_databox_without_boundary_conditions()));
      // Test the solver was applied to every tensor component in turn
      const auto& solver = minus_laplacian.solver();
      REQUIRE(solver.sources.size() == 3);
      check_component(solver.sources[0], source, ScalarFieldTag{}, 0);
      check_component(solver.sources[1], source, VectorFieldTag<Dim>{}, 0);
      check_component(solver.sources[2], source, VectorFieldTag<Dim>{}, 1);
      CHECK(solver.bc_types[0].empty());
      CHECK(solver.bc_types[1].empty());
      CHECK(solver.bc_types[2].empty());
      CHECK(minus_laplacian.cached_solvers().empty());
    }
    minus_laplacian.reset();
    {
      INFO("Subdomain with multiple boundary conditions");
      minus_laplacian.solve(
          make_not_null(&initial_guess_in_solution_out), linear_operator,
          source, std::make_tuple(make_databox_with_boundary_conditions()));
      // Test the solver was applied to every tensor component in turn
      // - A solver for each unique boundary-condition configuration should have
      //   been created and cached
      const auto& cached_solvers = minus_laplacian.cached_solvers();
      REQUIRE(cached_solvers.size() == 2);
      // - The solver for (D, N, D) should be used for the scalar
      const BcSignature signature_dnd{
          {{0, Direction<Dim>::upper_eta()},
           elliptic::BoundaryConditionType::Dirichlet},
          {{0, Direction<Dim>::lower_xi()},
           elliptic::BoundaryConditionType::Neumann},
          {{1, Direction<Dim>::lower_xi()},
           elliptic::BoundaryConditionType::Dirichlet}};
      REQUIRE(cached_solvers.at(signature_dnd).sources.size() == 1);
      check_component(cached_solvers.at(signature_dnd).sources[0], source,
                      ScalarFieldTag{}, 0);
      CHECK(cached_solvers.at(signature_dnd).bc_types[0] == signature_dnd);
      // - The solver for (D, D, N) should be used for both vector components
      const BcSignature signature_ddn{
          {{0, Direction<Dim>::upper_eta()},
           elliptic::BoundaryConditionType::Dirichlet},
          {{0, Direction<Dim>::lower_xi()},
           elliptic::BoundaryConditionType::Dirichlet},
          {{1, Direction<Dim>::lower_xi()},
           elliptic::BoundaryConditionType::Neumann}};
      REQUIRE(cached_solvers.at(signature_ddn).sources.size() == 2);
      check_component(cached_solvers.at(signature_ddn).sources[0], source,
                      VectorFieldTag<Dim>{}, 0);
      check_component(cached_solvers.at(signature_ddn).sources[1], source,
                      VectorFieldTag<Dim>{}, 1);
      CHECK(cached_solvers.at(signature_ddn).bc_types[0] == signature_ddn);
      CHECK(cached_solvers.at(signature_ddn).bc_types[1] == signature_ddn);
      // - The factory-constructed solver is not invoked, because it is only
      //   used as a template for each unique boundary-condition configuration
      CHECK(minus_laplacian.solver().sources.empty());
    }
    {
      INFO("Explicitly specified boundary condition");
      const auto minus_laplacian_dirichlet = TestHelpers::test_creation<
          MinusLaplacian<Dim, OptionsGroup, TestSolver<Dim>>>(
          "Solver:\n"
          "BoundaryConditions: Dirichlet\n");
      minus_laplacian_dirichlet.solve(
          make_not_null(&initial_guess_in_solution_out), linear_operator,
          source, std::make_tuple(make_databox_with_boundary_conditions()));
      // All boundary conditions are Dirichlet, so we should have only a single
      // cached solver
      const auto& cached_solvers = minus_laplacian_dirichlet.cached_solvers();
      REQUIRE(cached_solvers.size() == 1);
      // The solver for (D, D, D) should be used for all components
      const BcSignature signature_ddd{
          {{0, Direction<Dim>::upper_eta()},
           elliptic::BoundaryConditionType::Dirichlet},
          {{0, Direction<Dim>::lower_xi()},
           elliptic::BoundaryConditionType::Dirichlet},
          {{1, Direction<Dim>::lower_xi()},
           elliptic::BoundaryConditionType::Dirichlet}};
      const auto& cached_solver = cached_solvers.at(signature_ddd);
      REQUIRE(cached_solver.sources.size() == 3);
      check_component(cached_solver.sources[0], source, ScalarFieldTag{}, 0);
      check_component(cached_solver.sources[1], source, VectorFieldTag<Dim>{},
                      0);
      check_component(cached_solver.sources[2], source, VectorFieldTag<Dim>{},
                      1);
      CHECK(cached_solver.bc_types[0] == signature_ddd);
      CHECK(cached_solver.bc_types[1] == signature_ddd);
      CHECK(cached_solver.bc_types[2] == signature_ddd);
      CHECK(minus_laplacian_dirichlet.solver().sources.empty());
    }
  }
  {
    INFO("Factory-create the solver");
    constexpr size_t Dim = 2;
    using LinearSolverType = ::LinearSolver::Serial::LinearSolver<
        tmpl::list<Registrars::MinusLaplacian<Dim, OptionsGroup>>>;
    register_derived_classes_with_charm<LinearSolverType>();
    elliptic::subdomain_preconditioners::register_derived_with_charm();
    const auto created =
        TestHelpers::test_creation<std::unique_ptr<LinearSolverType>>(
            "MinusLaplacian:\n"
            "  Solver: ExplicitInverse\n"
            "  BoundaryConditions: Auto");
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
