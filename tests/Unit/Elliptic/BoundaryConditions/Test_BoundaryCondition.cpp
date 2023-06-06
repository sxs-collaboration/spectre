// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
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

class TestBoundaryCondition : public BoundaryCondition<1> {
 private:
  using Base = BoundaryCondition<1>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = "halp";

  TestBoundaryCondition() = default;
  TestBoundaryCondition(const TestBoundaryCondition&) = default;
  TestBoundaryCondition(TestBoundaryCondition&&) = default;
  TestBoundaryCondition& operator=(const TestBoundaryCondition&) = default;
  TestBoundaryCondition& operator=(TestBoundaryCondition&&) = default;
  ~TestBoundaryCondition() override = default;
  explicit TestBoundaryCondition(CkMigrateMessage* m) : Base(m) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestBoundaryCondition);
#pragma GCC diagnostic pop

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<TestBoundaryCondition>(*this);
  }

  using argument_tags =
      tmpl::list<ArgumentTag, VolumeArgumentTag, NonlinearArgumentTag>;
  using volume_tags = tmpl::list<VolumeArgumentTag>;

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {elliptic::BoundaryConditionType::Dirichlet};
  }

  // [example_poisson_fields]
  static void apply(const gsl::not_null<Scalar<DataVector>*> field,
                    const gsl::not_null<Scalar<DataVector>*> n_dot_flux,
                    const int arg_on_face, const bool arg_from_volume,
                    const int arg_nonlinear) {
    // [example_poisson_fields]
    CHECK(arg_on_face == 1);
    CHECK(arg_from_volume);
    CHECK(arg_nonlinear == 2);
    std::fill(field->begin(), field->end(), 1.);
    std::fill(n_dot_flux->begin(), n_dot_flux->end(), 2.);
  }

  using argument_tags_linearized = tmpl::list<ArgumentTag, VolumeArgumentTag>;
  using volume_tags_linearized = tmpl::list<VolumeArgumentTag>;

  static void apply_linearized(
      const gsl::not_null<Scalar<DataVector>*> field_correction,
      const gsl::not_null<Scalar<DataVector>*> n_dot_flux_correction,
      const int arg_on_face, const bool arg_from_volume) {
    CHECK(arg_on_face == 1);
    CHECK(arg_from_volume);
    std::fill(field_correction->begin(), field_correction->end(), 3.);
    std::fill(n_dot_flux_correction->begin(), n_dot_flux_correction->end(), 4.);
  }
};

PUP::able::PUP_ID TestBoundaryCondition::my_PUP_ID = 0;

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.BoundaryConditions.Base", "[Unit][Elliptic]") {
  // Factory-create a boundary condition and cast down to derived class
  const auto created = TestHelpers::test_factory_creation<
      BoundaryCondition<1>, TestBoundaryCondition>("TestBoundaryCondition");
  const auto& boundary_condition =
      dynamic_cast<const TestBoundaryCondition&>(*created);

  CHECK(created->boundary_condition_types() ==
        std::vector<elliptic::BoundaryConditionType>{
            elliptic::BoundaryConditionType::Dirichlet});

  // Test applying boundary conditions
  const auto box = db::create<
      db::AddSimpleTags<domain::Tags::Faces<1, ArgumentTag>, VolumeArgumentTag,
                        domain::Tags::Faces<1, NonlinearArgumentTag>>>(
      DirectionMap<1, int>{{Direction<1>::lower_xi(), 1}}, true,
      DirectionMap<1, int>{{Direction<1>::lower_xi(), 2}});
  const size_t num_points = 3;
  Scalar<DataVector> boundary_field{num_points};
  Scalar<DataVector> boundary_n_dot_flux{num_points};
  // Nonlinear
  elliptic::apply_boundary_condition<false, void,
                                     tmpl::list<TestBoundaryCondition>>(
      boundary_condition, box, Direction<1>::lower_xi(),
      make_not_null(&boundary_field), make_not_null(&boundary_n_dot_flux));
  CHECK(get(boundary_field) == DataVector{num_points, 1.});
  CHECK(get(boundary_n_dot_flux) == DataVector{num_points, 2.});
  // Linear
  elliptic::apply_boundary_condition<true, void,
                                     tmpl::list<TestBoundaryCondition>>(
      boundary_condition, box, Direction<1>::lower_xi(),
      make_not_null(&boundary_field), make_not_null(&boundary_n_dot_flux));
  CHECK(get(boundary_field) == DataVector{num_points, 3.});
  CHECK(get(boundary_n_dot_flux) == DataVector{num_points, 4.});
}

}  // namespace elliptic::BoundaryConditions
