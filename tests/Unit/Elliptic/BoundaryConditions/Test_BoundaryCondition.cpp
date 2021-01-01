// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::BoundaryConditions {

namespace {

struct ArgumentTag : db::SimpleTag {
  using type = int;
};

struct NonlinearArgumentTag : db::SimpleTag {
  using type = int;
};

struct VolumeArgumentTag : db::SimpleTag {
  using type = bool;
};

template <typename Registrars>
class TestBoundaryCondition;

struct TestRegistrar {
  template <typename Registrars>
  using f = TestBoundaryCondition<Registrars>;
};

template <typename Registrars = tmpl::list<TestRegistrar>>
class TestBoundaryCondition : public BoundaryCondition<1, Registrars> {
 private:
  using Base = BoundaryCondition<1, Registrars>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = "halp";

  TestBoundaryCondition() = default;
  TestBoundaryCondition(const TestBoundaryCondition&) = default;
  TestBoundaryCondition(TestBoundaryCondition&&) = default;
  TestBoundaryCondition& operator=(const TestBoundaryCondition&) = default;
  TestBoundaryCondition& operator=(TestBoundaryCondition&&) = default;
  ~TestBoundaryCondition() override = default;
  explicit TestBoundaryCondition(CkMigrateMessage* m) noexcept : Base(m) {}
  WRAPPED_PUPable_decl_base_template(
      SINGLE_ARG(BoundaryCondition<1, Registrars>),
      SINGLE_ARG(TestBoundaryCondition<Registrars>));  // NOLINT

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const noexcept override {
    return std::make_unique<TestBoundaryCondition>(*this);
  }

  using argument_tags =
      tmpl::list<ArgumentTag, VolumeArgumentTag, NonlinearArgumentTag>;
  using volume_tags = tmpl::list<VolumeArgumentTag>;

  /// [example_poisson_fields]
  static void apply(
      const int arg_on_face, const bool arg_from_volume,
      const int arg_nonlinear, const gsl::not_null<Scalar<DataVector>*> field,
      const gsl::not_null<Scalar<DataVector>*> n_dot_flux) noexcept {
    /// [example_poisson_fields]
    CHECK(arg_on_face == 1);
    CHECK(arg_from_volume);
    CHECK(arg_nonlinear == 2);
    std::fill(field->begin(), field->end(), 1.);
    std::fill(n_dot_flux->begin(), n_dot_flux->end(), 2.);
  }

  using argument_tags_linearized = tmpl::list<ArgumentTag, VolumeArgumentTag>;
  using volume_tags_linearized = tmpl::list<VolumeArgumentTag>;

  static void apply_linearized(
      const int arg_on_face, const bool arg_from_volume,
      const gsl::not_null<Scalar<DataVector>*> field_correction,
      const gsl::not_null<Scalar<DataVector>*> n_dot_flux_correction) noexcept {
    CHECK(arg_on_face == 1);
    CHECK(arg_from_volume);
    std::fill(field_correction->begin(), field_correction->end(), 3.);
    std::fill(n_dot_flux_correction->begin(), n_dot_flux_correction->end(), 4.);
  }
};

template <typename Registrars>
PUP::able::PUP_ID TestBoundaryCondition<Registrars>::my_PUP_ID = 0;

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.BoundaryConditions.Base", "[Unit][Elliptic]") {
  using BoundaryConditionType = TestBoundaryCondition<>;
  // Factory-create a boundary condition and cast down to derived class
  const auto boundary_condition_base = TestHelpers::test_factory_creation<
      BoundaryCondition<1, tmpl::list<TestRegistrar>>>("TestBoundaryCondition");
  const auto& boundary_condition =
      dynamic_cast<const BoundaryConditionType&>(*boundary_condition_base);

  // Test applying boundary conditions
  const auto box = db::create<db::AddSimpleTags<
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<1>,
                              ArgumentTag>,
      VolumeArgumentTag,
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<1>,
                              NonlinearArgumentTag>>>(
      std::unordered_map<Direction<1>, int>{{Direction<1>::lower_xi(), 1}},
      true,
      std::unordered_map<Direction<1>, int>{{Direction<1>::lower_xi(), 2}});
  const size_t num_points = 3;
  Scalar<DataVector> boundary_field{num_points};
  Scalar<DataVector> boundary_n_dot_flux{num_points};
  // Nonlinear
  elliptic::apply_boundary_condition<false>(
      boundary_condition, box, Direction<1>::lower_xi(),
      make_not_null(&boundary_field), make_not_null(&boundary_n_dot_flux));
  CHECK(get(boundary_field) == DataVector{num_points, 1.});
  CHECK(get(boundary_n_dot_flux) == DataVector{num_points, 2.});
  // Linear
  elliptic::apply_boundary_condition<true>(
      boundary_condition, box, Direction<1>::lower_xi(),
      make_not_null(&boundary_field), make_not_null(&boundary_n_dot_flux));
  CHECK(get(boundary_field) == DataVector{num_points, 3.});
  CHECK(get(boundary_n_dot_flux) == DataVector{num_points, 4.});
}

}  // namespace elliptic::BoundaryConditions
