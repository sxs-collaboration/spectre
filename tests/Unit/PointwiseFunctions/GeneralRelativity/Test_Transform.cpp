// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Transform.hpp"

namespace {

template <typename SrcTensorType, typename DestTensorType, typename DataType,
          size_t Dim, typename DestFrame, typename SrcFrame>
DestTensorType tensor_transformed_by_python(
    const SrcTensorType& src_tensor, const DestTensorType& /*dest_tensor*/,
    const ::Jacobian<DataType, Dim, DestFrame, SrcFrame>& jacobian,
    const std::string& suffix) {
  return pypp::call<DestTensorType>("Transform", "to_different_frame" + suffix,
                                    src_tensor, jacobian);
}

template <size_t Dim, typename SrcFrame, typename DestFrame, typename DataType>
void test_transform_to_different_frame(const DataType& used_for_size) {
  // Transform src->dest and then dest->src and ensure we recover
  // what we started with.
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> interval_dis_small(-0.1, 0.1);
  const auto nn_generator = make_not_null(&gen);
  const auto nn_interval_dis = make_not_null(&interval_dis);
  const auto nn_interval_dis_small = make_not_null(&interval_dis_small);

  const auto src_tensor =
      make_with_random_values<tnsr::ii<DataType, Dim, SrcFrame>>(
          nn_generator, nn_interval_dis, used_for_size);
  const auto jacobian = [&]() {
    auto jacobian_l =
        make_with_random_values<::Jacobian<DataType, Dim, DestFrame, SrcFrame>>(
            nn_generator, nn_interval_dis_small, used_for_size);
    // Ensure an invertible Jacobian by making the diagonal
    // elements 1+small, and all other elements small.
    for (size_t i = 0; i < Dim; ++i) {
      jacobian_l.get(i, i) += 1.0;
    }
    return jacobian_l;
  }();

  const auto dest_tensor = transform::to_different_frame(src_tensor, jacobian);
  const auto expected_src_tensor = transform::to_different_frame(
      dest_tensor, determinant_and_inverse(jacobian).second);
  CHECK_ITERABLE_APPROX(expected_src_tensor, src_tensor);

  // check transformation of tensor with python test
  CHECK_ITERABLE_APPROX(
      dest_tensor,
      tensor_transformed_by_python(src_tensor, dest_tensor, jacobian, ""));

  // check with different overloads
  const auto test_transform = [&jacobian, &nn_generator, &nn_interval_dis,
                               &used_for_size](auto transform_type,
                                               const std::string& suffix) {
    const auto src_tnsr = make_with_random_values<decltype(transform_type)>(
        nn_generator, nn_interval_dis, used_for_size);
    const auto dest_tnsr = transform::to_different_frame(
        src_tnsr, jacobian, determinant_and_inverse(jacobian).second);
    const auto dest_tnsr_py = tensor_transformed_by_python(
        src_tnsr, dest_tnsr, jacobian, "_" + suffix);
    CHECK_ITERABLE_APPROX(dest_tnsr, dest_tnsr_py);
  };
  test_transform(Scalar<DataType>{}, "Scalar");
  test_transform(tnsr::I<DataType, Dim, SrcFrame>{}, "I");
  test_transform(tnsr::i<DataType, Dim, SrcFrame>{}, "i");
  test_transform(tnsr::iJ<DataType, Dim, SrcFrame>{}, "iJ");
  test_transform(tnsr::ii<DataType, Dim, SrcFrame>{}, "ii");
  test_transform(tnsr::II<DataType, Dim, SrcFrame>{}, "II");
  test_transform(tnsr::ijj<DataType, Dim, SrcFrame>{}, "ijj");

  // special test case for the Scalar using not_null.
  const auto src_scalar = make_with_random_values<Scalar<DataType>>(
      nn_generator, nn_interval_dis, used_for_size);
  auto dest_scalar = make_with_value<Scalar<DataType>>(
      src_scalar, std::numeric_limits<double>::signaling_NaN());
  transform::to_different_frame(make_not_null(&dest_scalar), src_scalar,
                                jacobian,
                                determinant_and_inverse(jacobian).second);
  CHECK_ITERABLE_APPROX(dest_scalar, src_scalar);
}

template <size_t Dim, typename SrcFrame, typename DestFrame, typename DataType>
void test_transform_first_index_to_different_frame(
    const DataType& used_for_size) {
  tnsr::ijj<DataType, Dim, DestFrame> (*f)(
      const Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<Dim, UpLo::Lo, SrcFrame>,
                              SpatialIndex<Dim, UpLo::Lo, DestFrame>,
                              SpatialIndex<Dim, UpLo::Lo, DestFrame>>>&,
      const ::Jacobian<DataType, Dim, DestFrame, SrcFrame>&) =
      transform::first_index_to_different_frame<Dim, SrcFrame, DestFrame>;
  pypp::check_with_random_values<1>(f, "Transform",
                                    "first_index_to_different_frame",
                                    {{{-10., 10.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Transform",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");
  const DataVector dv(5);
  test_transform_to_different_frame<1, Frame::Grid, Frame::Inertial>(double{});
  test_transform_to_different_frame<2, Frame::Grid, Frame::Inertial>(double{});
  test_transform_to_different_frame<3, Frame::Grid, Frame::Inertial>(double{});
  test_transform_to_different_frame<1, Frame::Grid, Frame::Inertial>(dv);
  test_transform_to_different_frame<2, Frame::Grid, Frame::Inertial>(dv);
  test_transform_to_different_frame<3, Frame::Grid, Frame::Inertial>(dv);
  test_transform_to_different_frame<1, Frame::Inertial, Frame::Distorted>(
      double{});
  test_transform_to_different_frame<2, Frame::Inertial, Frame::Distorted>(
      double{});
  test_transform_to_different_frame<3, Frame::Inertial, Frame::Distorted>(
      double{});
  test_transform_to_different_frame<1, Frame::Inertial, Frame::Distorted>(dv);
  test_transform_to_different_frame<2, Frame::Inertial, Frame::Distorted>(dv);
  test_transform_to_different_frame<3, Frame::Inertial, Frame::Distorted>(dv);
  test_transform_first_index_to_different_frame<1, Frame::ElementLogical,
                                                Frame::Grid>(dv);
  test_transform_first_index_to_different_frame<2, Frame::ElementLogical,
                                                Frame::Grid>(dv);
  test_transform_first_index_to_different_frame<3, Frame::ElementLogical,
                                                Frame::Grid>(dv);
  test_transform_first_index_to_different_frame<1, Frame::ElementLogical,
                                                Frame::Distorted>(dv);
  test_transform_first_index_to_different_frame<2, Frame::ElementLogical,
                                                Frame::Distorted>(dv);
  test_transform_first_index_to_different_frame<3, Frame::ElementLogical,
                                                Frame::Distorted>(dv);
}
