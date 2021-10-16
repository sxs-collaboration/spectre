// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Sinusoid.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using InitialData = evolution::initial_data::InitialData;
using Sinusoid = ScalarAdvection::Solutions::Sinusoid;

void test_create() {
  const auto sine_wave = TestHelpers::test_creation<Sinusoid>("");
  CHECK(sine_wave == Sinusoid());
}

void test_serialize() {
  Sinusoid sine_wave;
  test_serialization(sine_wave);
}

void test_move() {
  Sinusoid sine_wave;
  Sinusoid sine_wave_copy;
  test_move_semantics(std::move(sine_wave), sine_wave_copy);  //  NOLINT
}

void test_derived() {
  Parallel::register_classes_with_charm<Sinusoid>();
  const std::unique_ptr<InitialData> initial_data_ptr =
      std::make_unique<Sinusoid>();
  const std::unique_ptr<InitialData> deserialized_initial_data_ptr =
      serialize_and_deserialize(initial_data_ptr);
  CHECK(dynamic_cast<Sinusoid*>(deserialized_initial_data_ptr.get()) !=
        nullptr);
}

struct SinusoidProxy : Sinusoid {
  using Sinusoid::Sinusoid;
  using variables_tags = tmpl::list<ScalarAdvection::Tags::U>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags> retrieve_variables(
      const tnsr::I<DataType, 1>& x, double t) const {
    return this->variables(x, t, variables_tags{});
  }
};

void verify_solution(const gsl::not_null<std::mt19937*> generator,
                     const size_t number_of_pts) {
  // DataVectors to store values
  Scalar<DataVector> u_sol{number_of_pts};
  Scalar<DataVector> u_test{number_of_pts};

  // generate random 1D grid points
  std::uniform_real_distribution<> distribution_coords(-1.0, 1.0);
  const auto x = make_with_random_values<tnsr::I<DataVector, 1>>(
      generator, make_not_null(&distribution_coords), u_sol);

  // test for random time
  std::uniform_real_distribution<> distribution_time(0.0, 10.0);
  double t{make_with_random_values<double>(generator,
                                           make_not_null(&distribution_time))};
  SinusoidProxy sine_wave;
  u_sol = get<ScalarAdvection::Tags::U>(sine_wave.retrieve_variables(x, t));
  u_test = pypp::call<Scalar<DataVector>>("Sinusoid", "u", x, t);
  CHECK_ITERABLE_APPROX(u_sol, u_test);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.ScalarAdvection.Sinusoid",
    "[Unit][PointwiseFunctions]") {
  test_create();
  test_serialize();
  test_move();
  test_derived();

  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/ScalarAdvection"};
  MAKE_GENERATOR(gen);
  verify_solution(make_not_null(&gen), 10);
}
