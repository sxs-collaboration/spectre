// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>

#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Options/Options.hpp"

namespace {
struct BoundaryCorrection;

struct BoundaryCorrectionBase {
  using creatable_classes = tmpl::list<BoundaryCorrection>;
  BoundaryCorrectionBase() = default;
  BoundaryCorrectionBase(const BoundaryCorrectionBase&) = default;
  BoundaryCorrectionBase& operator=(const BoundaryCorrectionBase&) = default;
  BoundaryCorrectionBase(BoundaryCorrectionBase&&) = default;
  BoundaryCorrectionBase& operator=(BoundaryCorrectionBase&&) = default;
  virtual ~BoundaryCorrectionBase() = 0;

  virtual std::unique_ptr<BoundaryCorrectionBase> get_clone()
      const noexcept = 0;
};

BoundaryCorrectionBase::~BoundaryCorrectionBase() = default;

struct BoundaryCorrection : public BoundaryCorrectionBase {
  BoundaryCorrection() = default;
  BoundaryCorrection(const BoundaryCorrection&) = default;
  BoundaryCorrection& operator=(const BoundaryCorrection&) = default;
  BoundaryCorrection(BoundaryCorrection&&) = default;
  BoundaryCorrection& operator=(BoundaryCorrection&&) = default;
  ~BoundaryCorrection() override = default;

  using options = tmpl::list<>;
  static constexpr Options::String help = {"Halp"};

  std::unique_ptr<BoundaryCorrectionBase> get_clone() const noexcept override {
    return std::make_unique<BoundaryCorrection>(*this);
  }
};

struct System {
  using boundary_correction_base = BoundaryCorrectionBase;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.BoundaryCorrectionTags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<evolution::Tags::BoundaryCorrection<System>>(
      "BoundaryCorrection");
  const auto boundary_correction = TestHelpers::test_creation<
      std::unique_ptr<BoundaryCorrectionBase>,
      evolution::OptionTags::BoundaryCorrection<System>>("BoundaryCorrection");
  CHECK(dynamic_cast<const BoundaryCorrection*>(boundary_correction.get()) !=
        nullptr);
  CHECK(evolution::Tags::BoundaryCorrection<System>::create_from_options(
            boundary_correction) != nullptr);
}
