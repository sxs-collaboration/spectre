// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/MomentumDensity.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"

namespace {
template <size_t SpatialDim, typename DataType>
void test_compute_item_in_databox(const DataType& used_for_size) {
  TestHelpers::db::test_compute_tag<
      ScalarWave::Tags::MomentumDensityCompute<SpatialDim>>("MomentumDensity");

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  const Scalar<DataVector> pi = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), dist, used_for_size);

  const tnsr::i<DataVector, SpatialDim, Frame::Inertial> phi =
      make_with_random_values<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), dist, used_for_size);

  const auto box = db::create<
      db::AddSimpleTags<ScalarWave::Tags::Pi,
                        ScalarWave::Tags::Phi<SpatialDim>>,
      db::AddComputeTags<ScalarWave::Tags::MomentumDensityCompute<SpatialDim>>>(
      pi, phi);

  const auto expected = ScalarWave::momentum_density(pi, phi);

  CHECK_ITERABLE_APPROX(
      (db::get<ScalarWave::Tags::MomentumDensity<SpatialDim>>(box)), expected);
}

template <size_t SpatialDim>
void test_momentum_density(const DataVector& used_for_size) {
  void (*f)(
      const gsl::not_null<tnsr::i<DataVector, SpatialDim, Frame::Inertial>*>,
      const Scalar<DataVector>&,
      const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&) =
      &ScalarWave::momentum_density<SpatialDim>;
  pypp::check_with_random_values<1>(f, "MomentumDensity", {"momentum_density"},
                                    {{{-1., 1.}}}, used_for_size);
  test_compute_item_in_databox<SpatialDim>(used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.MomentumDensity",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/ScalarWave/");

  const DataVector used_for_size(5);
  test_momentum_density<1>(used_for_size);
  test_momentum_density<2>(used_for_size);
  test_momentum_density<3>(used_for_size);
}
