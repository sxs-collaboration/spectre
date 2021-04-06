// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/GeneralRelativity/InterfaceNullNormal.hpp"

namespace {
template <size_t SpatialDim, typename DataType>
tnsr::A<DataType, SpatialDim, Frame::Inertial>
interface_outgoing_null_normal_one_form(
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_one_form,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_one_form) noexcept {
  return gr::interface_null_normal<SpatialDim, Frame::Inertial, DataType>(
      spacetime_normal_one_form, interface_normal_one_form, 1.);
}
template <size_t SpatialDim, typename DataType>
tnsr::A<DataType, SpatialDim, Frame::Inertial>
interface_incoming_null_normal_one_form(
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_one_form,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_one_form) noexcept {
  return gr::interface_null_normal<SpatialDim, Frame::Inertial, DataType>(
      spacetime_normal_one_form, interface_normal_one_form, -1.);
}
template <size_t SpatialDim, typename DataType>
tnsr::A<DataType, SpatialDim, Frame::Inertial>
interface_outgoing_null_normal_vector(
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_vector) noexcept {
  return gr::interface_null_normal<SpatialDim, Frame::Inertial, DataType>(
      spacetime_normal_vector, interface_normal_vector, 1.);
}
template <size_t SpatialDim, typename DataType>
tnsr::A<DataType, SpatialDim, Frame::Inertial>
interface_incoming_null_normal_vector(
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_vector) noexcept {
  return gr::interface_null_normal<SpatialDim, Frame::Inertial, DataType>(
      spacetime_normal_vector, interface_normal_vector, -1.);
}

template <size_t SpatialDim, typename DataType>
void test_interface_null_normals(const DataType& used_for_size) {
  {
    auto* f = &interface_outgoing_null_normal_one_form<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_outgoing_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
  {
    auto* f = &interface_outgoing_null_normal_vector<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_outgoing_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
  {
    auto* f = &interface_incoming_null_normal_one_form<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_incoming_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
  {
    auto* f = &interface_incoming_null_normal_vector<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_incoming_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.IntfcNullNormals",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_interface_null_normals, (1, 2, 3));
}
