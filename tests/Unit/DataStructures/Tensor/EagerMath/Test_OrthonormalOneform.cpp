// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/OrthonormalOneform.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/RandomUnitNormal.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <typename DataType, size_t Dim, typename Frame>
tnsr::ii<DataType, Dim, Frame> random_spatial_metric(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-0.05, 0.05);
  auto spatial_metric = make_with_random_values<tnsr::ii<DataType, Dim, Frame>>(
      generator, make_not_null(&distribution), used_for_size);
  for (size_t d = 0; d < Dim; ++d) {
    spatial_metric.get(d, d) += 1.0;
  }
  return spatial_metric;
}

template <size_t Dim>
struct TestOrthonormalForms {
  template <typename DataType, typename Frame>
  void operator()(
      const tnsr::i<DataType, Dim, Frame>& unit_form,
      const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
      const tnsr::II<DataType, Dim, Frame>& inv_spatial_metric) const noexcept;
};

template <>
struct TestOrthonormalForms<2> {
  template <typename DataType, typename Frame>
  void operator()(const tnsr::i<DataType, 2, Frame>& unit_form,
                  const tnsr::ii<DataType, 2, Frame>& /*spatial_metric*/,
                  const tnsr::II<DataType, 2, Frame>& inv_spatial_metric) const
      noexcept {
    const auto orthonormal_form =
        orthonormal_oneform(unit_form, inv_spatial_metric);

    const auto zero = make_with_value<Scalar<DataType>>(unit_form, 0.0);
    const auto one = make_with_value<Scalar<DataType>>(unit_form, 1.0);
    CHECK_ITERABLE_APPROX(dot_product(unit_form, unit_form, inv_spatial_metric),
                          one);
    CHECK_ITERABLE_APPROX(
        dot_product(orthonormal_form, orthonormal_form, inv_spatial_metric),
        one);
    CHECK_ITERABLE_APPROX(
        dot_product(unit_form, orthonormal_form, inv_spatial_metric), zero);
  }
};

template <>
struct TestOrthonormalForms<3> {
  template <typename DataType, typename Frame>
  void operator()(const tnsr::i<DataType, 3, Frame>& unit_form,
                  const tnsr::ii<DataType, 3, Frame>& spatial_metric,
                  const tnsr::II<DataType, 3, Frame>& inv_spatial_metric) const
      noexcept {
    const auto first_orthonormal_form =
        orthonormal_oneform(unit_form, inv_spatial_metric);
    const auto second_orthonormal_form =
        orthonormal_oneform(unit_form, first_orthonormal_form, spatial_metric,
                            determinant(spatial_metric));

    const auto zero = make_with_value<Scalar<DataType>>(unit_form, 0.0);
    const auto one = make_with_value<Scalar<DataType>>(unit_form, 1.0);
    CHECK_ITERABLE_APPROX(dot_product(unit_form, unit_form, inv_spatial_metric),
                          one);
    CHECK_ITERABLE_APPROX(
        dot_product(first_orthonormal_form, first_orthonormal_form,
                    inv_spatial_metric),
        one);
    CHECK_ITERABLE_APPROX(
        dot_product(second_orthonormal_form, second_orthonormal_form,
                    inv_spatial_metric),
        one);
    CHECK_ITERABLE_APPROX(
        dot_product(unit_form, first_orthonormal_form, inv_spatial_metric),
        zero);
    CHECK_ITERABLE_APPROX(
        dot_product(unit_form, second_orthonormal_form, inv_spatial_metric),
        zero);
    CHECK_ITERABLE_APPROX(
        dot_product(first_orthonormal_form, second_orthonormal_form,
                    inv_spatial_metric),
        zero);
  }
};

template <size_t Dim, typename Frame, typename DataType>
void check_orthonormal_forms(const DataType& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  const TestOrthonormalForms<Dim> test;

  const auto spatial_metric =
      random_spatial_metric<DataType, Dim, Frame>(&generator, used_for_size);
  const auto inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto unit_vector = random_unit_normal(&generator, spatial_metric);
  const auto unit_form = raise_or_lower_index(unit_vector, spatial_metric);

  // test for random unit form
  test(unit_form, spatial_metric, inv_spatial_metric);

  // test for unit form along coordinate axes
  for (const auto& direction : Direction<Dim>::all_directions()) {
    auto basis_form = euclidean_basis_vector(direction, used_for_size);
    const DataType inv_norm =
        1.0 / get(magnitude(basis_form, inv_spatial_metric));
    for (size_t i = 0; i < Dim; ++i) {
      basis_form.get(i) *= inv_norm;
    }
    test(basis_form, spatial_metric, inv_spatial_metric);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.OrthonormalOneform",
                  "[Unit][DataStructures]") {
  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(check_orthonormal_forms, (2, 3),
                                    (Frame::Inertial))
}
