// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <size_t Dim, typename Frame>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<
      ScalarTensor::Tags::TraceReversedStressEnergy<DataVector, Dim, Frame>>(
      "TraceReversedStressEnergy");
  TestHelpers::db::test_simple_tag<ScalarTensor::Tags::ScalarMass>(
      "ScalarMass");
  TestHelpers::db::test_simple_tag<ScalarTensor::Tags::ScalarSource>(
      "ScalarSource");
}

template <size_t Dim>
void test_prefix_tags() {
  TestHelpers::db::test_prefix_tag<
      ScalarTensor::Tags::Csw<CurvedScalarWave::Tags::Psi>>("Csw(Psi)");
  TestHelpers::db::test_prefix_tag<
      ScalarTensor::Tags::Csw<CurvedScalarWave::Tags::Pi>>("Csw(Pi)");
  TestHelpers::db::test_prefix_tag<
      ScalarTensor::Tags::Csw<CurvedScalarWave::Tags::Phi<Dim>>>("Csw(Phi)");
}

template <size_t Dim>
void test_compute_tags() {
  TestHelpers::db::test_compute_tag<
      ScalarTensor::Tags::CswCompute<CurvedScalarWave::Tags::Psi>>("Csw(Psi)");
  TestHelpers::db::test_compute_tag<
      ScalarTensor::Tags::CswCompute<CurvedScalarWave::Tags::Pi>>("Csw(Pi)");
  TestHelpers::db::test_compute_tag<
      ScalarTensor::Tags::CswCompute<CurvedScalarWave::Tags::Phi<Dim>>>(
      "Csw(Phi)");
  TestHelpers::db::test_compute_tag<
      ScalarTensor::Tags::CswOneIndexConstraintCompute<Dim>>(
      "Csw(OneIndexConstraint)");
  TestHelpers::db::test_compute_tag<
      ScalarTensor::Tags::CswTwoIndexConstraintCompute<Dim>>(
      "Csw(TwoIndexConstraint)");
}

template <size_t Dim>
void test_tags() {
  test_simple_tags<Dim, ArbitraryFrame>();
  test_prefix_tags<Dim>();
  test_compute_tags<Dim>();
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarTensor.Tags",
                  "[Unit][Evolution]") {
  test_tags<1_st>();
  test_tags<2_st>();
  test_tags<3_st>();
  TestHelpers::test_option_tag<ScalarTensor::OptionTags::ScalarMass>("1.0");
}
