// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/LaneEmdenStar.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

void test_construction_and_serialization() {
  const double central_mass_density = 0.7;
  const double polytropic_constant = 2.0;
  NewtonianEuler::Solutions::LaneEmdenStar star(central_mass_density,
                                                polytropic_constant);

  const std::string input =
      MakeString{} << "  CentralMassDensity: " << central_mass_density
                   << "\n  PolytropicConstant: " << polytropic_constant;
  const auto star_from_options =
      TestHelpers::test_creation<NewtonianEuler::Solutions::LaneEmdenStar>(
          input);

  CHECK(star_from_options == star);

  NewtonianEuler::Solutions::LaneEmdenStar star_to_move(central_mass_density,
                                                        polytropic_constant);
  test_move_semantics(std::move(star_to_move), star);  //  NOLINT

  test_serialization(star);
}

struct LaneEmdenStarProxy : NewtonianEuler::Solutions::LaneEmdenStar {
  using NewtonianEuler::Solutions::LaneEmdenStar::LaneEmdenStar;

  template <typename DataType>
  using variables_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensity<DataType>,
                 NewtonianEuler::Tags::Velocity<DataType, 3, Frame::Inertial>,
                 NewtonianEuler::Tags::SpecificInternalEnergy<DataType>,
                 NewtonianEuler::Tags::Pressure<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>>
  primitive_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                      double t) const noexcept {
    return this->variables(x, t, variables_tags<DataType>{});
  }
};

template <typename DataType>
void test_solution(const DataType& used_for_size,
                   const double central_mass_density,
                   const double polytropic_constant) noexcept {
  const LaneEmdenStarProxy star(central_mass_density, polytropic_constant);
  pypp::check_with_random_values<
      1, typename LaneEmdenStarProxy::template variables_tags<DataType>>(
      &LaneEmdenStarProxy::template primitive_variables<DataType>, star,
      "LaneEmdenStar",
      {"mass_density", "velocity", "specific_internal_energy", "pressure"},
      // with polytropic_constant == 2, star has outer radius ~ 1.77
      {{{-2.0, 2.0}}},
      std::make_tuple(central_mass_density, polytropic_constant),
      used_for_size);

  const auto star_sd = serialize_and_deserialize(star);
  pypp::check_with_random_values<
      1, typename LaneEmdenStarProxy::template variables_tags<DataType>>(
      &LaneEmdenStarProxy::template primitive_variables<DataType>, star_sd,
      "LaneEmdenStar",
      {"mass_density", "velocity", "specific_internal_energy", "pressure"},
      // with polytropic_constant == 2, star has outer radius ~ 1.77
      {{{-2.0, 2.0}}},
      std::make_tuple(central_mass_density, polytropic_constant),
      used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.LaneEmdenStar",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/NewtonianEuler"};

  test_construction_and_serialization();

  // Nothing special about these values, we just want them to be non-unity and
  // to be different from each other:
  const double central_mass_density = 0.7;
  const double polytropic_constant = 2.0;
  test_solution(std::numeric_limits<double>::signaling_NaN(),
                central_mass_density, polytropic_constant);
  test_solution(DataVector(5), central_mass_density, polytropic_constant);
}
