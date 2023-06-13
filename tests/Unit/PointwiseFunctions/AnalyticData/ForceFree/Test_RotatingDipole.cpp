// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/RotatingDipole.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using InitialData = evolution::initial_data::InitialData;
using RotatingDipole = ForceFree::AnalyticData::RotatingDipole;

struct RotatingDipoleProxy : RotatingDipole {
  using RotatingDipole::RotatingDipole;
  using variables_tags =
      tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                 ForceFree::Tags::TildePsi, ForceFree::Tags::TildePhi,
                 ForceFree::Tags::TildeQ>;

  tuples::tagged_tuple_from_typelist<variables_tags> return_variables(
      const tnsr::I<DataVector, 3>& x) const {
    return this->variables(x, variables_tags{});
  }
};

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.ForceFree.RotatingDipole",
    "[Unit][PointwiseFunctions]") {
  // test creation
  const auto solution = TestHelpers::test_creation<RotatingDipole>(
      "VectorPotentialAmplitude: 1.0\n"
      "Varpi0 : 0.5\n"
      "Delta : 0.1\n"
      "AngularVelocity : 0.3\n"
      "TiltAngle : 0.0");
  CHECK(solution == RotatingDipole(1.0, 0.5, 0.1, 0.3, 0.0));
  CHECK(solution != RotatingDipole(2.0, 0.5, 0.1, 0.3, 0.0));
  CHECK(solution != RotatingDipole(1.0, 1.0, 0.1, 0.3, 0.0));
  CHECK(solution != RotatingDipole(1.0, 0.5, 1.0, 0.3, 0.0));
  CHECK(solution != RotatingDipole(1.0, 0.5, 0.1, 0.5, 0.0));
  CHECK(solution != RotatingDipole(1.0, 0.5, 0.1, 0.3, 1.0));

  CHECK_THROWS_WITH(
      []() { const RotatingDipole soln(1.0, -0.5, 0.1, 0.3, 0.0); }(),
      Catch::Matchers::ContainsSubstring("The length constant varpi0"));

  CHECK_THROWS_WITH(
      []() { const RotatingDipole soln(1.0, 0.5, -0.1, 0.3, 0.0); }(),
      Catch::Matchers::ContainsSubstring("The small number delta"));

  CHECK_THROWS_WITH(
      []() { const RotatingDipole soln(1.0, 0.5, 0.1, 2.0, 0.0); }(),
      Catch::Matchers::ContainsSubstring("must be between -1.0 and 1.0"));

  CHECK_THROWS_WITH(
      []() { const RotatingDipole soln(1.0, 0.5, 0.1, 0.3, 7.0); }(),
      Catch::Matchers::ContainsSubstring("must be between 0 and Pi"));

  // test serialize
  test_serialization(solution);

  // test move
  test_move_semantics(RotatingDipole{1.0, 0.5, 0.1, 0.3, 0.0},
                      RotatingDipole{1.0, 0.5, 0.1, 0.3, 0.0});

  // test derived
  register_classes_with_charm<RotatingDipole>();
  const std::unique_ptr<InitialData> base_ptr =
      std::make_unique<RotatingDipole>(1.0, 0.5, 0.1, 0.3, 0.0);
  const std::unique_ptr<InitialData> deserialized_base_ptr =
      serialize_and_deserialize(base_ptr)->get_clone();
  CHECK(dynamic_cast<const RotatingDipole&>(*deserialized_base_ptr.get()) ==
        dynamic_cast<const RotatingDipole&>(*base_ptr.get()));

  // test solution
  const DataVector used_for_size{10};

  const double vector_potential_amplitude = 1.1;
  const double varpi0 = 0.3;
  const double delta = 0.07;
  const double angular_velocity = 0.123;
  const double tilt_angle = 0.456;
  const auto member_variables = std::make_tuple(
      vector_potential_amplitude, varpi0, delta, angular_velocity, tilt_angle);

  RotatingDipoleProxy rotating_dipole(vector_potential_amplitude, varpi0, delta,
                                      angular_velocity, tilt_angle);

  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/ForceFree"};

  // Check for the EM variables
  pypp::check_with_random_values<1>(
      &RotatingDipoleProxy::return_variables, rotating_dipole, "RotatingDipole",
      {"TildeE", "TildeB", "TildePsi", "TildePhi", "TildeQ"}, {{{-10.0, 10.0}}},
      member_variables, used_for_size);

  // Check the member function `angular_velocity()`
  CHECK(solution.angular_velocity() == 0.3);

  // Check the member function `interior_mask()`
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto random_coords =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, used_for_size);
  const Scalar<DataVector> mask_from_python{pypp::call<Scalar<DataVector>>(
      "RotatingDipole", "InteriorMask", random_coords)};
  CHECK(solution.interior_mask(random_coords) == mask_from_python);
}
}  // namespace
