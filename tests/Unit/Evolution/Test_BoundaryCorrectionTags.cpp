// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>

#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Options/String.hpp"

namespace {
struct BoundaryCorrection : public evolution::BoundaryCorrection {
  BoundaryCorrection() = default;
  BoundaryCorrection(const BoundaryCorrection&) = default;
  BoundaryCorrection& operator=(const BoundaryCorrection&) = default;
  BoundaryCorrection(BoundaryCorrection&&) = default;
  BoundaryCorrection& operator=(BoundaryCorrection&&) = default;
  ~BoundaryCorrection() override = default;

  using options = tmpl::list<>;
  static constexpr Options::String help = {"Halp"};

  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(BoundaryCorrection);  // NOLINT
#pragma GCC diagnostic pop

  std::unique_ptr<evolution::BoundaryCorrection> get_clone() const override {
    return std::make_unique<BoundaryCorrection>(*this);
  }
};

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID BoundaryCorrection::my_PUP_ID = 0;  // NOLINT
#endif                                                // SPECTRE_USE_CHARM
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.BoundaryCorrectionTags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<evolution::Tags::BoundaryCorrection>(
      "BoundaryCorrection");
  const auto boundary_correction =
      TestHelpers::test_option_tag_factory_creation<
          evolution::OptionTags::BoundaryCorrection, BoundaryCorrection>(
          "BoundaryCorrection");
  CHECK(dynamic_cast<const BoundaryCorrection*>(boundary_correction.get()) !=
        nullptr);
  CHECK(evolution::Tags::BoundaryCorrection::create_from_options(
            boundary_correction) != nullptr);
}
