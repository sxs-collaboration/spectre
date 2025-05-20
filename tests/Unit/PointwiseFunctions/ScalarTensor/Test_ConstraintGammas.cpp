// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/ConstraintDamping/Constant.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"
#include "PointwiseFunctions/ScalarTensor/ConstraintDampingTags.hpp"
#include "PointwiseFunctions/ScalarTensor/ConstraintGammas.hpp"
#include "Time/Tags/Time.hpp"

namespace {
struct ArbitraryFrame;

template <size_t Dim, typename Fr>
void test_tags() {
  TestHelpers::db::test_compute_tag<
      ScalarTensor::Tags::ConstraintGamma1Compute<Dim, Fr>>("ConstraintGamma1");
  TestHelpers::db::test_compute_tag<
      ScalarTensor::Tags::ConstraintGamma2Compute<Dim, Fr>>("ConstraintGamma2");
}

void test_tag_retrieval() {
  const size_t num_points = 4;
  const size_t volume_dim = 3;
  const double initial_time = 0.0;
  const double expiration_time = 2.0;
  const double value1 = 1.0;
  const double value2 = 2.0;

  using simple_tags = db::AddSimpleTags<
      domain::Tags::Coordinates<volume_dim, Frame::Grid>, ::Tags::Time,
      ::domain::Tags::FunctionsOfTimeInitialize,
      ScalarTensor::Tags::DampingFunctionGamma1<volume_dim, Frame::Grid>,
      ScalarTensor::Tags::DampingFunctionGamma2<volume_dim, Frame::Grid>>;
  using compute_tags = db::AddComputeTags<
      ScalarTensor::Tags::ConstraintGamma1Compute<volume_dim, Frame::Grid>,
      ScalarTensor::Tags::ConstraintGamma2Compute<volume_dim, Frame::Grid>>;

  const tnsr::I<DataVector, volume_dim, Frame::Grid> grid_coords =
      make_with_value<tnsr::I<DataVector, volume_dim, Frame::Grid>>(num_points,
                                                                    1.0);
  const std::array<DataVector, 3> init_func{{{0.0}, {0.0}, {2.0}}};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["translation"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, init_func, expiration_time);

  auto box = db::create<simple_tags, compute_tags>(
      grid_coords, 1.0, std::move(functions_of_time),
      std::unique_ptr<
          ConstraintDamping::DampingFunction<volume_dim, Frame::Grid>>{
          std::make_unique<
              ConstraintDamping::Constant<volume_dim, Frame::Grid>>(value1)},
      std::unique_ptr<
          ConstraintDamping::DampingFunction<volume_dim, Frame::Grid>>{
          std::make_unique<
              ConstraintDamping::Constant<volume_dim, Frame::Grid>>(value2)});

  const auto& retrieved_gamma1 =
      get<CurvedScalarWave::Tags::ConstraintGamma1>(box);
  const auto& retrieved_gamma2 =
      get<CurvedScalarWave::Tags::ConstraintGamma2>(box);

  const Scalar<DataVector> expected_gamma1 =
      make_with_value<Scalar<DataVector>>(num_points, value1);
  const Scalar<DataVector> expected_gamma2 =
      make_with_value<Scalar<DataVector>>(num_points, value2);

  CHECK_ITERABLE_APPROX(retrieved_gamma1, expected_gamma1);
  CHECK_ITERABLE_APPROX(retrieved_gamma2, expected_gamma2);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.ConstraintGammas",
                  "[Unit][PointwiseFunctions]") {
  test_tags<1, ArbitraryFrame>();
  test_tags<2, ArbitraryFrame>();
  test_tags<3, ArbitraryFrame>();
  test_tag_retrieval();
}
