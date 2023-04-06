// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// [criterion_examples]
struct FieldOne : db::SimpleTag {
  using type = double;
};

struct FieldTwo : db::SimpleTag {
  using type = double;
};

struct Constraint : db::SimpleTag {
  using type = double;
};

struct ConstraintCompute : db::ComputeTag, Constraint {
  using base = Constraint;
  using return_type = double;
  using argument_tags = tmpl::list<FieldOne, FieldTwo>;
  static void function(const gsl::not_null<double*> result,
                       const double field_one, const double field_two) {
    *result = field_one - field_two;
  }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
class CriterionOne : public amr::Criterion {
 public:
  struct CriticalValue {
    using type = double;
    static constexpr Options::String help = {
        "The critical value of field one ."};
  };
  using options = tmpl::list<CriticalValue>;

  static constexpr Options::String help = {
      "h-refine the grid if field one is above a critical value"};

  CriterionOne() = default;
  explicit CriterionOne(const double critical_value)
      : critical_value_(critical_value) {}
  explicit CriterionOne(CkMigrateMessage* msg) : Criterion(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CriterionOne);  // NOLINT

  using compute_tags_for_observartion_box = tmpl::list<>;
  using argument_tags = tmpl::list<FieldOne>;

  template <typename Metavariables>
  auto operator()(
      const double field_one, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Metavariables::volume_dim>& /*element_id*/) const {
    return field_one > critical_value_ ? std::array{amr::Flag::Split}
                                       : std::array{amr::Flag::DoNothing};
  }

  void pup(PUP::er& p) override {
    Criterion::pup(p);
    p | critical_value_;
  }

 private:
  double critical_value_{0.0};
};

PUP::able::PUP_ID CriterionOne::my_PUP_ID = 0;  // NOLINT

class CriterionTwo : public amr::Criterion {
 public:
  struct TargetValue {
    using type = double;
    static constexpr Options::String help = {"The target value."};
  };
  using options = tmpl::list<TargetValue>;

  static constexpr Options::String help = {
      "h-refine if the absolute value of the constraint is above the target "
      "value.  h-coarsen if the constraint is an order of magnitude below the "
      "target value"};

  CriterionTwo() = default;
  explicit CriterionTwo(const double target_value)
      : target_value_(target_value) {}
  explicit CriterionTwo(CkMigrateMessage* /*msg*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CriterionTwo);  // NOLINT

  using compute_tags_for_observartion_box = tmpl::list<ConstraintCompute>;
  using argument_tags = tmpl::list<Constraint>;

  template <typename Metavariables>
  auto operator()(
      const double constraint, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Metavariables::volume_dim>& /*element_id*/) const {
    return std::abs(constraint) > target_value_
               ? std::array{amr::Flag::Split}
               : (std::abs(constraint) < 0.1 * target_value_
                      ? std::array{amr::Flag::Join}
                      : std::array{amr::Flag::DoNothing});
  }

  void pup(PUP::er& p) override {
    Criterion::pup(p);
    p | target_value_;
  }

 private:
  double target_value_{0.0};
};

PUP::able::PUP_ID CriterionTwo::my_PUP_ID = 0;  // NOLINT
#pragma GCC diagnostic pop

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<amr::Criterion, tmpl::list<CriterionOne, CriterionTwo>>>;
  };
};
// [criterion_examples]

void test_criterion(const amr::Criterion& criterion, const double field_one,
                    const double field_two, const amr::Flag expected_flag) {
  Parallel::GlobalCache<Metavariables> empty_cache{};
  using simple_tags = tmpl::list<FieldOne, FieldTwo>;
  const auto databox = db::create<simple_tags>(field_one, field_two);
  // This list is the union of all compute_tags_for_observation_box for all
  // criteria listed in Metavariables::factory_creation::factory_classes
  // It can be constructed with a metafunction, but for this simple test
  // we just explicitly list them
  using compute_tags = tmpl::list<ConstraintCompute>;
  ObservationBox<compute_tags, db::DataBox<simple_tags>> box{databox};
  ElementId<1> element_id{0};
  auto flags = criterion.evaluate(box, empty_cache, element_id);
  CHECK(flags == std::array{expected_flag});
}

void test() {
  register_factory_classes_with_charm<Metavariables>();
  const CriterionOne one{1.0};
  test_criterion(one, 2.0, 0.0, amr::Flag::Split);

  test_criterion(serialize_and_deserialize(one), 0.5, 0.0,
                 amr::Flag::DoNothing);
  const auto one_option =
      TestHelpers::test_creation<std::unique_ptr<amr::Criterion>,
                                 Metavariables>(
          "CriterionOne:\n"
          "  CriticalValue: 3.0\n");
  test_criterion(*one_option, 2.0, 0.0, amr::Flag::DoNothing);
  test_criterion(*serialize_and_deserialize(one_option), 2.0, 0.0,
                 amr::Flag::DoNothing);
  const CriterionTwo two{1.e-6};
  test_criterion(two, 4.e-6, 6.e-6, amr::Flag::Split);
  test_criterion(serialize_and_deserialize(two), 4.e-6, 4.5e-6,
                 amr::Flag::DoNothing);
  const auto two_option =
      TestHelpers::test_creation<std::unique_ptr<amr::Criterion>,
                                 Metavariables>(
          "CriterionTwo:\n"
          "  TargetValue: 1.e-5\n");
  test_criterion(*two_option, 4.e-7, 3.e-7, amr::Flag::Join);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Criteria.Criterion", "[Unit][ParallelAlgorithms]") {
  test();
}
