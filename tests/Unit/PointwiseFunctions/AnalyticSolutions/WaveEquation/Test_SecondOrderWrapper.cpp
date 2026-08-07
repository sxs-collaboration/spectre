// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SecondOrderWrapper.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

// Tests SecondOrderWrapper as an adapter around
// ScalarWave::Solutions::PlaneWave. The wrapped solution's mathematics (psi,
// dpsi_dt, ...) is exercised by Test_PlaneWave; here we only verify the adapter
// contract: the second-order `variables` interfaces re-expose the wrapped
// solution's values under the SecondOrderScalarWave tags.
namespace {
// Tag namespaces of the wrapped (first-order) solution and the second-order
// system the adapter targets.
namespace sw_tags = ScalarWave::Tags;
namespace so_tags = SecondOrderScalarWave::Tags;

template <size_t Dim>
using Wrapper = SecondOrderScalarWave::Solutions::SecondOrderWrapper<
    ScalarWave::Solutions::PlaneWave<Dim>>;

template <size_t Dim>
std::string yaml_sequence(const std::array<double, Dim>& values) {
  std::stringstream result{};
  result << std::setprecision(std::numeric_limits<double>::max_digits10) << "[";
  for (size_t d = 0; d < Dim; ++d) {
    if (d > 0) {
      result << ", ";
    }
    result << gsl::at(values, d);
  }
  result << "]";
  return result.str();
}

template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   tmpl::list<MathFunctions::PowX<1, Frame::Inertial>>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::list<Wrapper<Dim>>>>;
  };
};

// Verifies the three second-order `variables` interfaces against the
// independently supplied first-order PlaneWave. This establishes that the
// plane-wave values are preserved under the mapping to the
// SecondOrderScalarWave tags.
template <size_t Dim>
void check_adapter_contract(
    const Wrapper<Dim>& second_order_plane_wave,
    const ScalarWave::Solutions::PlaneWave<Dim>& plane_wave,
    const tnsr::I<DataVector, Dim>& x, const double t) {
  const auto plane_wave_vars = plane_wave.variables(
      x, t, tmpl::list<sw_tags::Psi, sw_tags::Pi, sw_tags::Phi<Dim>>{});

  // Evolved and auxiliary variables {Psi, Pi, Phi}: the values from the wrapped
  // solution are preserved under the tag mapping.
  const auto second_order_vars = second_order_plane_wave.variables(
      x, t, tmpl::list<so_tags::Psi, so_tags::Pi, so_tags::Phi<Dim>>{});
  CHECK(get<so_tags::Psi>(second_order_vars) ==
        get<sw_tags::Psi>(plane_wave_vars));
  CHECK(get<so_tags::Pi>(second_order_vars) ==
        get<sw_tags::Pi>(plane_wave_vars));
  CHECK(get<so_tags::Phi<Dim>>(second_order_vars) ==
        get<sw_tags::Phi<Dim>>(plane_wave_vars));

  // Evolved variables {Psi, Pi}.
  const auto second_order_evolved_vars = second_order_plane_wave.variables(
      x, t, tmpl::list<so_tags::Psi, so_tags::Pi>{});
  static_assert(
      std::is_same_v<std::decay_t<decltype(second_order_evolved_vars)>,
                     tuples::TaggedTuple<so_tags::Psi, so_tags::Pi>>,
      "The evolved-variables overload must return exactly {Psi, Pi}.");
  CHECK(get<so_tags::Psi>(second_order_evolved_vars) ==
        get<sw_tags::Psi>(plane_wave_vars));
  CHECK(get<so_tags::Pi>(second_order_evolved_vars) ==
        get<sw_tags::Pi>(plane_wave_vars));

  // Time derivatives of the evolved variables {dt<Psi>, dt<Pi>}: the return
  // type drops the wrapped solution's dt<Phi>.
  const auto plane_wave_dt_vars = plane_wave.variables(
      x, t,
      tmpl::list<Tags::dt<sw_tags::Psi>, Tags::dt<sw_tags::Pi>,
                 Tags::dt<sw_tags::Phi<Dim>>>{});
  const auto second_order_dt_vars = second_order_plane_wave.variables(
      x, t, tmpl::list<Tags::dt<so_tags::Psi>, Tags::dt<so_tags::Pi>>{});
  static_assert(
      std::is_same_v<
          std::decay_t<decltype(second_order_dt_vars)>,
          tuples::TaggedTuple<Tags::dt<so_tags::Psi>, Tags::dt<so_tags::Pi>>>,
      "The second-order dt overload must drop dt<Phi>.");
  CHECK(get<Tags::dt<so_tags::Psi>>(second_order_dt_vars) ==
        get<Tags::dt<sw_tags::Psi>>(plane_wave_dt_vars));
  CHECK(get<Tags::dt<so_tags::Pi>>(second_order_dt_vars) ==
        get<Tags::dt<sw_tags::Pi>>(plane_wave_dt_vars));
}

template <size_t Dim>
void test_wrapper(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> wave_magnitude_distribution(0.5, 2.0);
  std::uniform_real_distribution<> value_distribution(-2.0, 2.0);
  auto wave_vector = make_with_random_values<std::array<double, Dim>>(
      generator, make_not_null(&wave_magnitude_distribution),
      std::array<double, Dim>{});
  for (size_t d = 1; d < Dim; d += 2) {
    gsl::at(wave_vector, d) *= -1.0;
  }
  const auto center = make_with_random_values<std::array<double, Dim>>(
      generator, make_not_null(&value_distribution), std::array<double, Dim>{});
  const auto make_profile = [](const int power) {
    return std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(power);
  };

  const ScalarWave::Solutions::PlaneWave<Dim> plane_wave(wave_vector, center,
                                                         make_profile(3));
  const SecondOrderScalarWave::Solutions::SecondOrderWrapper
      second_order_plane_wave(plane_wave);
  static_assert(
      std::is_same_v<std::remove_const_t<decltype(second_order_plane_wave)>,
                     Wrapper<Dim>>,
      "Class template argument deduction must yield Wrapper<Dim>.");
  // The wrapper holds the ScalarWave solution by composition, not inheritance,
  // so the wrapped interface stays hidden.
  static_assert(not std::is_base_of_v<ScalarWave::Solutions::PlaneWave<Dim>,
                                      Wrapper<Dim>>,
                "The wrapper must not inherit the wrapped solution.");
  // Differs only in the profile, exercising inequality through the wrapped
  // solution's operator==.
  const Wrapper<Dim> different_second_order_plane_wave(wave_vector, center,
                                                       make_profile(2));

  const auto x = make_with_random_values<tnsr::I<DataVector, Dim>>(
      generator, make_not_null(&value_distribution), DataVector(5));
  const auto t = make_with_random_values<double>(
      generator, make_not_null(&value_distribution));

  // The user-facing factory name of a wrapped PlaneWave is unchanged.
  CHECK(pretty_type::name<Wrapper<Dim>>() == "SecondOrderPlaneWave");

  // Equality and inequality.
  CHECK(second_order_plane_wave == second_order_plane_wave);
  CHECK_FALSE(second_order_plane_wave != second_order_plane_wave);
  CHECK(second_order_plane_wave != different_second_order_plane_wave);
  CHECK_FALSE(second_order_plane_wave == different_second_order_plane_wave);

  // Copy and move semantics.
  test_copy_semantics(second_order_plane_wave);
  auto second_order_plane_wave_to_move = second_order_plane_wave;
  test_move_semantics(std::move(second_order_plane_wave_to_move),
                      second_order_plane_wave);

  // The adapter contract for the three second-order `variables` interfaces,
  // compared against the independently accessible wrapped solution.
  check_adapter_contract(second_order_plane_wave, plane_wave, x, t);

  // Serialization round-trip preserves equality and the adapter contract.
  register_factory_classes_with_charm<Metavariables<Dim>>();
  const auto deserialized_second_order_plane_wave =
      serialize_and_deserialize(second_order_plane_wave);
  CHECK(deserialized_second_order_plane_wave == second_order_plane_wave);
  check_adapter_contract(deserialized_second_order_plane_wave, plane_wave, x,
                         t);

  // Clone behavior.
  const auto cloned_initial_data = second_order_plane_wave.get_clone();
  CHECK(dynamic_cast<const Wrapper<Dim>&>(*cloned_initial_data) ==
        second_order_plane_wave);

  // Factory creation from options through the SecondOrderPlaneWave name, and
  // serialization of the option-created solution.
  const std::string option_string =
      "SecondOrderPlaneWave:\n"
      "  WaveVector: " +
      yaml_sequence(wave_vector) + "\n  Center: " + yaml_sequence(center) +
      "\n  Profile:\n"
      "    PowX:\n"
      "      Power: 3";
  const std::unique_ptr<evolution::initial_data::InitialData>
      option_created_initial_data = TestHelpers::test_option_tag<
          evolution::initial_data::OptionTags::InitialData, Metavariables<Dim>>(
          option_string);
  CHECK(dynamic_cast<const Wrapper<Dim>&>(*option_created_initial_data) ==
        second_order_plane_wave);
  const auto deserialized_option_created_initial_data =
      serialize_and_deserialize(option_created_initial_data);
  CHECK(dynamic_cast<const Wrapper<Dim>&>(
            *deserialized_option_created_initial_data) ==
        second_order_plane_wave);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.WaveEquation."
    "SecondOrderWrapper",
    "[PointwiseFunctions][Unit]") {
  MAKE_GENERATOR(generator);
  test_wrapper<1>(make_not_null(&generator));
  test_wrapper<2>(make_not_null(&generator));
  test_wrapper<3>(make_not_null(&generator));
}
