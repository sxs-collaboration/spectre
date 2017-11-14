// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/range/combine.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <typename Symmetry, typename IndexList>
void check_tensor_doubles_equals_tensor_datavectors(
    const Tensor<DataVector, Symmetry, IndexList>& tensor_dv,
    const Tensor<double, Symmetry, IndexList>& tensor_double) {
  const size_t n_pts = tensor_dv.begin()->size();
  for (decltype(auto) datavector_and_double_components :
       boost::combine(tensor_dv, tensor_double)) {
    for (size_t s = 0; s < n_pts; ++s) {
      CHECK(boost::get<0>(datavector_and_double_components)[s] ==
            boost::get<1>(datavector_and_double_components));
    }
  }
}

template <typename DataType>
auto make_lapse(const DataType& structure) {
  return Scalar<DataType>{make_with_value<DataType>(structure, 3.)};
}

template <typename DataType>
auto make_shift(const DataType& structure) {
  tnsr::I<DataType, 3> shift{};
  for (size_t i = 0; i < 3; ++i) {
    shift.get(i) = make_with_value<DataType>(structure, i);
  }
  return shift;
}

template <typename DataType>
auto make_spatial_metric(const DataType& structure) {
  tnsr::ii<DataType, 3> metric{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      metric.get(i, j) =
          make_with_value<DataType>(structure, (i + 1.) * (j + 1.));
    }
  }
  return metric;
}

void test_compute_spacetime_metric(const DataVector& structure) {
  const auto psi = compute_spacetime_metric(make_lapse(0.), make_shift(0.),
                                            make_spatial_metric(0.));
  CHECK(psi.get(0, 0) == approx(55.));
  CHECK(psi.get(0, 1) == approx(8.));
  CHECK(psi.get(0, 2) == approx(16.));
  CHECK(psi.get(0, 3) == approx(24.));
  CHECK(psi.get(1, 1) == approx(1.));
  CHECK(psi.get(1, 2) == approx(2.));
  CHECK(psi.get(1, 3) == approx(3.));
  CHECK(psi.get(2, 2) == approx(4.));
  CHECK(psi.get(2, 3) == approx(6.));
  CHECK(psi.get(3, 3) == approx(9.));

  check_tensor_doubles_equals_tensor_datavectors(
      compute_spacetime_metric(make_lapse(structure),
                               make_shift(structure),
                               make_spatial_metric(structure)),
      psi);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrFunctions.SpacetimeDecomp",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv(2);
  test_compute_spacetime_metric(dv);
}
