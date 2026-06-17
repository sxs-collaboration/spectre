// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/SpectralFilter.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using Vars = Variables<tmpl::list<ScalarFieldTag>>;
using FilterBase = Filters::Filter<1, tmpl::list<ScalarFieldTag>>;

// Minimal concrete filter for testing the action.
//
// apply_in_volume multiplies each component of the Variables by 2 and records
// whether the optional Jacobians were populated.
class MockFilter : public FilterBase {
 public:
  MockFilter() = default;
  MockFilter(bool apply_substep, bool apply_this_step, bool need_jacs,
             bool supports_mesh = true)
      : apply_substep_(apply_substep),
        apply_this_step_(apply_this_step),
        need_jacs_(need_jacs),
        supports_mesh_(supports_mesh) {}

  explicit MockFilter(CkMigrateMessage* m) : FilterBase(m) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  // NOLINTNEXTLINE
  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(FilterBase), MockFilter);
#pragma GCC diagnostic pop

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    FilterBase::pup(p);
    p | apply_substep_;
    p | apply_this_step_;
    p | need_jacs_;
    p | supports_mesh_;
    p | saw_inv_jac_;
    p | saw_jac_;
  }

  std::unique_ptr<FilterBase> get_clone() const override {
    return std::make_unique<MockFilter>(*this);
  }

  bool apply_volume_filter_on_substep() const override {
    return apply_substep_;
  }
  bool apply_volume_filter_on_this_step(size_t /*step_number*/) const override {
    return apply_this_step_;
  }
  bool apply_boundary_filter_on_substep() const override { return false; }
  bool apply_boundary_filter_on_this_step(
      size_t /*step_number*/) const override {
    return false;
  }
  bool need_jacobians() const override { return need_jacs_; }
  bool supports_mesh(const Mesh<1>& /*mesh*/) const override {
    return supports_mesh_;
  }
  std::string name() const override { return "MockFilter"; }
  bool is_equal(const FilterBase& other) const override {
    const auto* rhs = dynamic_cast<const MockFilter*>(&other);
    return rhs != nullptr and apply_substep_ == rhs->apply_substep_ and
           apply_this_step_ == rhs->apply_this_step_ and
           need_jacs_ == rhs->need_jacs_;
  }

  const std::optional<std::vector<size_t>>& blocks_to_filter() const override {
    return blocks_to_filter_;
  }
  void set_blocks_to_filter(
      const std::vector<std::string>& /*all_block_names*/,
      const std::unordered_map<std::string, std::unordered_set<std::string>>&
      /*block_groups*/) override {}

  void apply_in_volume(
      gsl::not_null<Vars*> vars, const Mesh<1>& /*mesh*/,
      const std::optional<InverseJacobian<DataVector, 1, Frame::Grid,
                                          Frame::Inertial>>& inv_jac,
      const std::optional<Jacobian<DataVector, 1, Frame::Grid,
                                   Frame::Inertial>>& jac) const override {
    saw_inv_jac_ = inv_jac.has_value();
    saw_jac_ = jac.has_value();
    for (auto& component : get<ScalarFieldTag>(*vars)) {
      component *= 2.0;
    }
  }

  void apply_on_boundary(
      gsl::not_null<Vars*> /*vars*/, const Mesh<0>& /*mesh*/,
      const std::optional<
          InverseJacobian<DataVector, 1, Frame::Grid, Frame::Inertial>>&
      /*inv_jac*/,
      const std::optional<
          Jacobian<DataVector, 1, Frame::Grid, Frame::Inertial>>&
      /*jac*/) const override {}

  bool saw_inv_jac() const { return saw_inv_jac_; }
  bool saw_jac() const { return saw_jac_; }

 private:
  bool apply_substep_{false};
  bool apply_this_step_{false};
  bool need_jacs_{false};
  bool supports_mesh_{true};
  std::optional<std::vector<size_t>> blocks_to_filter_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable bool saw_inv_jac_{false};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable bool saw_jac_{false};
};

// NOLINTNEXTLINE
PUP::able::PUP_ID MockFilter::my_PUP_ID = 0;

using VarsTag = ::Tags::Variables<tmpl::list<ScalarFieldTag>>;
using InvJacTag =
    domain::Tags::InverseJacobian<1, Frame::Grid, Frame::Inertial>;
using JacTag = domain::Tags::Jacobian<1, Frame::Grid, Frame::Inertial>;
using FilterTag = Filters::Tags::SpectralFilter<1, tmpl::list<ScalarFieldTag>>;
using StepTag = ::Tags::StepNumberWithinSlab;
using MeshTag = domain::Tags::Mesh<1>;
using ElementTag = domain::Tags::Element<1>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<VarsTag, MeshTag, ElementTag, StepTag, FilterTag,
                         InvJacTag, JacTag>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<dg::Actions::SpectralFilter>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tags = tmpl::list<>;

  struct system {
    static constexpr size_t volume_dim = 1;
    using variables_tag = VarsTag;
  };
};

InverseJacobian<DataVector, 1, Frame::Grid, Frame::Inertial>
make_identity_inv_jac() {
  return InverseJacobian<DataVector, 1, Frame::Grid, Frame::Inertial>{5_st,
                                                                      1.0};
}

Jacobian<DataVector, 1, Frame::Grid, Frame::Inertial> make_identity_jac() {
  return Jacobian<DataVector, 1, Frame::Grid, Frame::Inertial>{5_st, 1.0};
}

template <typename Filter, typename Callback>
void run_action(std::unique_ptr<Filter> filter_ptr, const double initial_value,
                const uint64_t step_number, Callback&& callback) {
  using element_array = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, 0,
      {Vars(5, initial_value),
       Mesh<1>{5, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto},
       Element<1>{ElementId<1>{0}, {}}, step_number,
       std::unique_ptr<FilterBase>(std::move(filter_ptr)),
       make_identity_inv_jac(), make_identity_jac()});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), 0);
  std::forward<Callback>(callback)(runner);
}

DataVector get_scalar_field(
    const ActionTesting::MockRuntimeSystem<Metavariables>& runner) {
  return get(get<ScalarFieldTag>(
      ActionTesting::get_databox_tag<ElementArray<Metavariables>, VarsTag>(
          runner, 0)));
}

const MockFilter* get_mock_filter(
    const ActionTesting::MockRuntimeSystem<Metavariables>& runner) {
  const auto& filter_ref =
      ActionTesting::get_databox_tag<ElementArray<Metavariables>, FilterTag>(
          runner, 0);
  return dynamic_cast<const MockFilter*>(&filter_ref);
}

void test_none_filter_skips() {
  run_action(std::make_unique<Filters::None<1, tmpl::list<ScalarFieldTag>>>(
                 std::nullopt),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 3.0));
             });
}

// Test: MockFilter with both gating flags false does not apply.
void test_no_gating_flags_skips() {
  run_action(std::make_unique<MockFilter>(/*apply_substep=*/false,
                                          /*apply_this_step=*/false,
                                          /*need_jacs=*/false),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 3.0));
             });
}

// Test: apply_volume_filter_on_substep=true → filter applies.
void test_substep_flag_applies_filter() {
  run_action(std::make_unique<MockFilter>(/*apply_substep=*/true,
                                          /*apply_this_step=*/false,
                                          /*need_jacs=*/false),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 6.0));
             });
}

// Test: apply_volume_filter_on_this_step=true → filter applies.
void test_step_flag_applies_filter() {
  run_action(std::make_unique<MockFilter>(/*apply_substep=*/false,
                                          /*apply_this_step=*/true,
                                          /*need_jacs=*/false),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 6.0));
             });
}

// Test: need_jacobians=true → Jacobians populated in apply_in_volume.
void test_jacobians_passed_when_needed() {
  run_action(std::make_unique<MockFilter>(/*apply_substep=*/true,
                                          /*apply_this_step=*/false,
                                          /*need_jacs=*/true),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 6.0));
               const MockFilter* mock = get_mock_filter(runner);
               REQUIRE(mock != nullptr);
               CHECK(mock->saw_inv_jac());
               CHECK(mock->saw_jac());
             });
}

// Test: need_jacobians=false → Jacobians are nullopt in apply_in_volume.
void test_jacobians_not_passed_when_not_needed() {
  run_action(std::make_unique<MockFilter>(/*apply_substep=*/true,
                                          /*apply_this_step=*/false,
                                          /*need_jacs=*/false),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 6.0));
               const MockFilter* mock = get_mock_filter(runner);
               REQUIRE(mock != nullptr);
               CHECK(not mock->saw_inv_jac());
               CHECK(not mock->saw_jac());
             });
}

// Test: a filter that does not support the mesh triggers an ERROR naming the
// filter.
void test_unsupported_mesh_errors() {
  CHECK_THROWS_WITH(
      run_action(std::make_unique<MockFilter>(/*apply_substep=*/true,
                                              /*apply_this_step=*/false,
                                              /*need_jacs=*/false,
                                              /*supports_mesh=*/false),
                 3.0, 0, [](const auto& /*runner*/) {}),
      Catch::Matchers::ContainsSubstring("MockFilter") and
          Catch::Matchers::ContainsSubstring("does not support"));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Actions.SpectralFilter",
                  "[Unit][ParallelAlgorithms][Actions]") {
  register_classes_with_charm<MockFilter,
                              Filters::None<1, tmpl::list<ScalarFieldTag>>>();
  test_none_filter_skips();
  test_no_gating_flags_skips();
  test_substep_flag_applies_filter();
  test_step_flag_applies_filter();
  test_jacobians_passed_when_needed();
  test_jacobians_not_passed_when_not_needed();
  test_unsupported_mesh_errors();
}
