// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct FactorOne : db::SimpleTag {
  using type = double;
};

struct FactorTwo : db::SimpleTag {
  using type = double;
};

struct Product : db::SimpleTag {
  using type = double;
};

struct ProductCompute : db::ComputeTag, Product {
  using base = Product;
  using return_type = double;
  using argument_tags = tmpl::list<FactorOne, FactorTwo>;

  static void function(const gsl::not_null<double*> result,
                       const double field_one, const double field_two) {
    *result = field_one * field_two;
  }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
class CriterionOne : public ah::Criterion {
 public:
  struct TargetValue {
    using type = double;
    static constexpr Options::String help = {"The target value."};
  };
  using options = tmpl::list<TargetValue>;

  static constexpr Options::String help = {
      "Increase surface resolution if the factor one is above a target value; "
      "otherwise decreases the resolution"};

  CriterionOne() = default;
  explicit CriterionOne(const double target_value)
      : target_value_(target_value) {}
  explicit CriterionOne(CkMigrateMessage* /*msg*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CriterionOne);  // NOLINT

  std::string observation_name() override { return "CriterionOne"; }

  using compute_tags_for_observartion_box = tmpl::list<>;
  using argument_tags = tmpl::list<FactorOne>;

  template <typename Metavariables, typename Fr>
  size_t operator()(const double field_one,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ylm::Strahlkorper<Fr>& strahlkorper,
                    const FastFlow::IterInfo& /*info*/) const {
    return field_one > target_value_ ? strahlkorper.l_max() + 1
                                     : strahlkorper.l_max() - 1;
  }

  void pup(PUP::er& p) override {
    Criterion::pup(p);
    p | target_value_;
  }

 private:
  double target_value_{std::numeric_limits<double>::signaling_NaN()};
};

PUP::able::PUP_ID CriterionOne::my_PUP_ID = 0;  // NOLINT

class CriterionTwo : public ah::Criterion {
 public:
  struct TargetValue {
    using type = double;
    static constexpr Options::String help = {"The target value."};
  };
  using options = tmpl::list<TargetValue>;

  static constexpr Options::String help = {
      "Increase surface resolution if the product of the two factors times the "
      "(simulated) residual of a surface find is above "
      "a target value; decrease resolution otherwise."};

  CriterionTwo() = default;
  explicit CriterionTwo(const double target_value)
      : target_value_(target_value) {}
  explicit CriterionTwo(CkMigrateMessage* /*msg*/) {}
  using PUP::able::register_constructor;        // NOLINT
  WRAPPED_PUPable_decl_template(CriterionTwo);  // NOLINT

  std::string observation_name() override { return "CriterionTwo"; }

  using compute_tags_for_observartion_box = tmpl::list<ProductCompute>;
  using argument_tags = tmpl::list<Product>;

  template <typename Metavariables, typename Fr>
  size_t operator()(const double product,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ylm::Strahlkorper<Fr>& strahlkorper,
                    const FastFlow::IterInfo& info) const {
    return product * info.max_residual > target_value_
               ? strahlkorper.l_max() + 1
               : strahlkorper.l_max() - 1;
  }

  void pup(PUP::er& p) override {
    Criterion::pup(p);
    p | target_value_;
  }

 private:
  double target_value_{std::numeric_limits<double>::signaling_NaN()};
};

PUP::able::PUP_ID CriterionTwo::my_PUP_ID = 0;  // NOLINT
#pragma GCC diagnostic pop

struct Metavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<ah::Criterion, tmpl::list<CriterionOne, CriterionTwo>>>;
  };
};

template <typename Fr>
void test_criterion(const ah::Criterion& criterion, const double field_one,
                    const double field_two, const size_t expected_l_max,
                    const ylm::Strahlkorper<Fr>& strahlkorper,
                    const FastFlow::IterInfo& info) {
  Parallel::GlobalCache<Metavariables> empty_cache{};
  using simple_tags = tmpl::list<FactorOne, FactorTwo>;
  auto databox = db::create<simple_tags>(field_one, field_two);
  using compute_tags = tmpl::list<ProductCompute>;
  const auto box = make_observation_box<compute_tags>(make_not_null(&databox));

  auto new_l_max = criterion.evaluate(box, empty_cache, strahlkorper, info);
  CHECK(new_l_max == expected_l_max);
}

void test() {
  register_factory_classes_with_charm<Metavariables>();
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{
      11, 4.5, {{0.1, 0.2, 0.3}}};
  FastFlow::IterInfo info{};
  info.max_residual = 1.e-6;

  const CriterionOne criterion_one{2.3};
  test_criterion(criterion_one, 3.4, 0.0, strahlkorper.l_max() + 1,
                 strahlkorper, info);
  test_criterion(serialize_and_deserialize(criterion_one), 3.4, 0.0,
                 strahlkorper.l_max() + 1, strahlkorper, info);
  const auto criterion_one_option =
      TestHelpers::test_creation<std::unique_ptr<ah::Criterion>, Metavariables>(
          "CriterionOne:\n"
          "  TargetValue: 2.3\n");
  test_criterion(*criterion_one_option, 1.8, 0.0, strahlkorper.l_max() - 1,
                 strahlkorper, info);
  test_criterion(*serialize_and_deserialize(criterion_one_option), 2.2, 0.0,
                 strahlkorper.l_max() - 1, strahlkorper, info);

  const CriterionTwo criterion_two{4.0e-6};
  test_criterion(criterion_two, 2.0, 1.5, strahlkorper.l_max() - 1,
                 strahlkorper, info);
  test_criterion(serialize_and_deserialize(criterion_two), 2.0, 1.5,
                 strahlkorper.l_max() - 1, strahlkorper, info);
  const auto criterion_two_option =
      TestHelpers::test_creation<std::unique_ptr<ah::Criterion>, Metavariables>(
          "CriterionTwo:\n"
          "  TargetValue: 4.0e-6\n");
  test_criterion(*criterion_two_option, 2.0, 1.5, strahlkorper.l_max() - 1,
                 strahlkorper, info);
  test_criterion(*serialize_and_deserialize(criterion_two_option), 2.0, 1.5,
                 strahlkorper.l_max() - 1, strahlkorper, info);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.Criteria.Criterion",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
