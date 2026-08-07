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
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/StandingWave.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

// Test the first-order ScalarWave::Solutions::StandingWave against a pair of
// counter-propagating ScalarWave::Solutions::PlaneWave objects whose sum is the
// standing wave.  Both the subject and the truth use the ScalarWave tags.
namespace {
namespace sw_tags = ScalarWave::Tags;

std::string yaml_double(const double value) {
  std::stringstream result{};
  result << std::setprecision(std::numeric_limits<double>::max_digits10)
         << value;
  return result.str();
}

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
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::list<ScalarWave::Solutions::StandingWave<Dim>>>>;
  };
};

// Compares every field the standing wave exposes against the sum of the
// corresponding fields of the right- and left-moving first-order plane waves.
// The plane-wave truth is obtained through the same complete variables
// interfaces used on the subject, so all six fields (Psi, Pi, Phi and their
// time derivatives) are exercised generically.
template <size_t Dim>
void check_against_truth(
    const ScalarWave::Solutions::StandingWave<Dim>& subject,
    const ScalarWave::Solutions::PlaneWave<Dim>& right_mover,
    const ScalarWave::Solutions::PlaneWave<Dim>& left_mover,
    const tnsr::I<DataVector, Dim>& x, const double t) {
  using variables_tags =
      tmpl::list<sw_tags::Psi, sw_tags::Pi, sw_tags::Phi<Dim>>;
  using dt_variables_tags =
      tmpl::list<::Tags::dt<sw_tags::Psi>, ::Tags::dt<sw_tags::Pi>,
                 ::Tags::dt<sw_tags::Phi<Dim>>>;

  // The full first-order truth {Psi, Pi, Phi} of each mover.
  const auto right_vars = right_mover.variables(x, t, variables_tags{});
  const auto left_vars = left_mover.variables(x, t, variables_tags{});

  const Scalar<DataVector> truth_psi(get(get<sw_tags::Psi>(right_vars)) +
                                     get(get<sw_tags::Psi>(left_vars)));
  const Scalar<DataVector> truth_pi(get(get<sw_tags::Pi>(right_vars)) +
                                    get(get<sw_tags::Pi>(left_vars)));
  tnsr::i<DataVector, Dim> truth_phi = get<sw_tags::Phi<Dim>>(right_vars);
  for (size_t d = 0; d < Dim; ++d) {
    truth_phi.get(d) += get<sw_tags::Phi<Dim>>(left_vars).get(d);
  }

  const auto subject_vars = subject.variables(x, t, variables_tags{});
  CHECK_ITERABLE_APPROX(get<sw_tags::Psi>(subject_vars), truth_psi);
  CHECK_ITERABLE_APPROX(get<sw_tags::Pi>(subject_vars), truth_pi);
  CHECK_ITERABLE_APPROX(get<sw_tags::Phi<Dim>>(subject_vars), truth_phi);

  // The full first-order time-derivative truth {dt Psi, dt Pi, dt Phi} of each
  // mover, from the same generic interface.
  const auto right_dt_vars = right_mover.variables(x, t, dt_variables_tags{});
  const auto left_dt_vars = left_mover.variables(x, t, dt_variables_tags{});

  const Scalar<DataVector> truth_dt_psi(
      get(get<::Tags::dt<sw_tags::Psi>>(right_dt_vars)) +
      get(get<::Tags::dt<sw_tags::Psi>>(left_dt_vars)));
  const Scalar<DataVector> truth_dt_pi(
      get(get<::Tags::dt<sw_tags::Pi>>(right_dt_vars)) +
      get(get<::Tags::dt<sw_tags::Pi>>(left_dt_vars)));
  tnsr::i<DataVector, Dim> truth_dt_phi =
      get<::Tags::dt<sw_tags::Phi<Dim>>>(right_dt_vars);
  for (size_t d = 0; d < Dim; ++d) {
    truth_dt_phi.get(d) +=
        get<::Tags::dt<sw_tags::Phi<Dim>>>(left_dt_vars).get(d);
  }

  const auto subject_dt_vars = subject.variables(x, t, dt_variables_tags{});
  CHECK_ITERABLE_APPROX(get<::Tags::dt<sw_tags::Psi>>(subject_dt_vars),
                        truth_dt_psi);
  CHECK_ITERABLE_APPROX(get<::Tags::dt<sw_tags::Pi>>(subject_dt_vars),
                        truth_dt_pi);
  CHECK_ITERABLE_APPROX(get<::Tags::dt<sw_tags::Phi<Dim>>>(subject_dt_vars),
                        truth_dt_phi);
}

template <size_t Dim>
void test_standing_wave(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> magnitude_distribution(0.5, 2.0);
  std::uniform_real_distribution<> value_distribution(-2.0, 2.0);
  auto wave_vector = make_with_random_values<std::array<double, Dim>>(
      generator, make_not_null(&magnitude_distribution),
      std::array<double, Dim>{});
  for (size_t d = 1; d < Dim; d += 2) {
    gsl::at(wave_vector, d) *= -1.0;
  }
  const auto center = make_with_random_values<std::array<double, Dim>>(
      generator, make_not_null(&value_distribution), std::array<double, Dim>{});
  const auto amplitude = make_with_random_values<double>(
      generator, make_not_null(&magnitude_distribution));

  const ScalarWave::Solutions::StandingWave<Dim> subject(wave_vector, center,
                                                         amplitude);

  // Decompose Psi = A sin(k.(x-x0)) cos(|k| t) into two plane
  // waves using sin(a) cos(b) = 1/2 [sin(a - b) + sin(a + b)]:
  //   right-mover: wave vector +k, profile  (A/2) sin(u) with
  //                u =  k.(x-x0) - |k| t  ->  (A/2) sin(k.(x-x0) - |k| t)
  //   left-mover:  wave vector -k, profile -(A/2) sin(u) with
  //                u = -k.(x-x0) - |k| t  ->  (A/2) sin(k.(x-x0) + |k| t)
  // Both share the standing wave's center; |k| is unchanged by negating k.
  auto negated_wave_vector = wave_vector;
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(negated_wave_vector, d) *= -1.0;
  }
  const ScalarWave::Solutions::PlaneWave<Dim> right_mover(
      wave_vector, center,
      std::make_unique<MathFunctions::Sinusoid<1, Frame::Inertial>>(
          0.5 * amplitude, 1.0, 0.0));
  const ScalarWave::Solutions::PlaneWave<Dim> left_mover(
      negated_wave_vector, center,
      std::make_unique<MathFunctions::Sinusoid<1, Frame::Inertial>>(
          -0.5 * amplitude, 1.0, 0.0));

  const auto x = make_with_random_values<tnsr::I<DataVector, Dim>>(
      generator, make_not_null(&value_distribution), DataVector(5));
  const auto t = make_with_random_values<double>(
      generator, make_not_null(&value_distribution));

  CHECK_FALSE(subject != subject);
  test_copy_semantics(subject);
  auto subject_to_move = subject;
  test_move_semantics(std::move(subject_to_move), subject);
  check_against_truth(subject, right_mover, left_mover, x, t);
  // At t = 0 the counter-propagating movers have equal and opposite Pi, so the
  // standing wave has Pi = 0; the truth sum reproduces this without a formula.
  check_against_truth(subject, right_mover, left_mover, x, 0.0);

  // Direct serialization round-trip.
  register_factory_classes_with_charm<Metavariables<Dim>>();
  const auto deserialized_subject = serialize_and_deserialize(subject);
  CHECK(deserialized_subject == subject);
  check_against_truth(deserialized_subject, right_mover, left_mover, x, t);

  // Clone behavior.
  const auto clone = subject.get_clone();
  CHECK(dynamic_cast<const ScalarWave::Solutions::StandingWave<Dim>&>(*clone) ==
        subject);

  // Factory creation from options, and serialization of the option-created
  // solution.  The YAML is generated from the same random values at
  // max_digits10 precision so exact object equality remains valid.
  const std::string option_string =
      "StandingWave:\n"
      "  WaveVector: " +
      yaml_sequence(wave_vector) + "\n  Center: " + yaml_sequence(center) +
      "\n  Amplitude: " + yaml_double(amplitude);
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag<
          evolution::initial_data::OptionTags::InitialData, Metavariables<Dim>>(
          option_string);
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& created_solution =
      dynamic_cast<const ScalarWave::Solutions::StandingWave<Dim>&>(
          *deserialized_option_solution);
  CHECK(created_solution == subject);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.WaveEquation.StandingWave",
    "[PointwiseFunctions][Unit]") {
  MAKE_GENERATOR(generator);
  test_standing_wave<1>(make_not_null(&generator));
  test_standing_wave<2>(make_not_null(&generator));
  test_standing_wave<3>(make_not_null(&generator));
}
