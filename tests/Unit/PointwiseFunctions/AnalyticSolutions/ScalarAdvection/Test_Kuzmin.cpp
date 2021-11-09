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
#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Kuzmin.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using InitialData = evolution::initial_data::InitialData;
using Kuzmin = ScalarAdvection::Solutions::Kuzmin;

void test_create() {
  const auto sol = TestHelpers::test_creation<Kuzmin>("");
  CHECK(sol == Kuzmin());
}

void test_serialize() {
  Kuzmin sol;
  test_serialization(sol);
}

void test_move() {
  Kuzmin sol;
  Kuzmin sol_copy;
  test_move_semantics(std::move(sol), sol_copy);  //  NOLINT
}

void test_derived() {
  Parallel::register_classes_with_charm<Kuzmin>();
  const std::unique_ptr<InitialData> initial_data_ptr =
      std::make_unique<Kuzmin>();
  const std::unique_ptr<InitialData> deserialized_initial_data_ptr =
      serialize_and_deserialize(initial_data_ptr);
  CHECK(dynamic_cast<Kuzmin*>(deserialized_initial_data_ptr.get()) != nullptr);
}

struct KuzminProxy : Kuzmin {
  using Kuzmin::Kuzmin;
  using variables_tags = tmpl::list<ScalarAdvection::Tags::U>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags> retrieve_variables(
      const tnsr::I<DataType, 2>& x, double t) const {
    return this->variables(x, t, variables_tags{});
  }
};

void verify_solution(const gsl::not_null<std::mt19937*> generator,
                     const size_t number_of_pts_per_region) {
  // we need to distribute random points for 5 different regions (see the for
  // loop below)
  const size_t num_pts = 5 * number_of_pts_per_region;

  // DataVectors to store values
  Scalar<DataVector> u_sol{num_pts};
  Scalar<DataVector> u_test{num_pts};
  auto coords_init = make_with_value<tnsr::I<DataVector, 2>>(
      5 * number_of_pts_per_region, 0.0);
  auto& x0 = get<0>(coords_init);
  auto& y0 = get<1>(coords_init);

  // random displacements within a circle of radius 0.15
  std::uniform_real_distribution<> distribution_radius(0.0, 0.15);
  std::uniform_real_distribution<> distribution_angle(0.0, 2 * M_PI);
  const auto radius = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution_radius), number_of_pts_per_region);
  const auto angle = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution_angle), number_of_pts_per_region);
  auto x_circle =
      make_with_value<tnsr::I<DataVector, 2>>(number_of_pts_per_region, 0.0);
  auto& xc = get<0>(x_circle);
  auto& yc = get<1>(x_circle);
  xc = get(radius) * cos(get(angle));
  yc = get(radius) * sin(get(angle));

  // random displacements within a square [0, 1] x [0, 1]
  std::uniform_real_distribution<> distribution_xy(0.0, 1.0);
  const auto x_square = make_with_random_values<tnsr::I<DataVector, 2>>(
      generator, make_not_null(&distribution_xy), number_of_pts_per_region);

  for (size_t i = 0; i < number_of_pts_per_region; ++i) {
    // a cylinder centered at (0.5, 0.75)
    x0[i] = xc[i] + 0.5;
    y0[i] = yc[i] + 0.75;
    // and the slot inside the cylinder
    x0[number_of_pts_per_region + i] = 0.05 * get<0>(x_square)[i] + 0.475;
    y0[number_of_pts_per_region + i] = 0.3 * get<1>(x_square)[i] + 0.6;
    // cone centered at (0.5, 0.25)
    x0[2 * number_of_pts_per_region + i] = xc[i] + 0.5;
    y0[2 * number_of_pts_per_region + i] = yc[i] + 0.25;
    // hump centered at (0.25, 0.5)
    x0[3 * number_of_pts_per_region + i] = xc[i] + 0.25;
    y0[3 * number_of_pts_per_region + i] = yc[i] + 0.5;
    // other regions with u=0; some of pts may overlap with previous ones
    x0[4 * number_of_pts_per_region + i] = get<0>(x_square)[i];
    y0[4 * number_of_pts_per_region + i] = get<1>(x_square)[i];
  }

  // generate random time and shift coordinates
  std::uniform_real_distribution<> distribution_time(0.0, 10.0);
  double t{make_with_random_values<double>(generator,
                                           make_not_null(&distribution_time))};
  auto x = make_with_value<tnsr::I<DataVector, 2>>(num_pts, 0.0);
  get<0>(x) = (x0 - 0.5) * cos(t) - (y0 - 0.5) * sin(t) + 0.5;
  get<1>(x) = (x0 - 0.5) * sin(t) + (y0 - 0.5) * cos(t) + 0.5;

  // test the solution
  KuzminProxy sol;
  u_sol = get<ScalarAdvection::Tags::U>(sol.retrieve_variables(x, t));
  u_test = pypp::call<Scalar<DataVector>>("Kuzmin", "u", x, t);
  CHECK_ITERABLE_APPROX(u_sol, u_test);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.ScalarAdvection.Kuzmin",
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
