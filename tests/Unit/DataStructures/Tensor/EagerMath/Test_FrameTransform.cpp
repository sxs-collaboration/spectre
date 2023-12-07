// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/FrameTransform.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables/FrameTransform.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"

namespace transform {
namespace {

struct Var1 : db::SimpleTag {
  using type = tnsr::I<DataVector, 2, Frame::Inertial>;
};
struct Var2 : db::SimpleTag {
  using type = tnsr::Ij<DataVector, 2, Frame::Inertial>;
};

template <typename SrcTensorType, typename DestTensorType, typename DataType,
          size_t Dim, typename DestFrame, typename SrcFrame>
DestTensorType tensor_transformed_by_python(
    const SrcTensorType& src_tensor, const DestTensorType& /*dest_tensor*/,
    const ::Jacobian<DataType, Dim, DestFrame, SrcFrame>& jacobian,
    const std::string& suffix) {
  return pypp::call<DestTensorType>(
      "FrameTransform", "to_different_frame" + suffix, src_tensor, jacobian);
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

void test_transform_first_index_to_different_frame() {
  INFO("Transform first index");
  {
    INFO("1D");
    InverseJacobian<double, 1, Frame::ElementLogical, Frame::Inertial>
        inv_jacobian{};
    get<0, 0>(inv_jacobian) = 2.0;
    {
      INFO("Vector");
      const tnsr::I<double, 1, Frame::Inertial> input{1.0};
      const auto result = first_index_to_different_frame(input, inv_jacobian);
      static_assert(std::is_same_v<std::decay_t<decltype(result)>,
                                   tnsr::I<double, 1, Frame::ElementLogical>>);
      CHECK(get<0>(result) == 2.0);
    }
    {
      INFO("Rank 2 tensor");
      tnsr::Ij<double, 1, Frame::Inertial> input{};
      get<0, 0>(input) = 1.0;
      const auto result = first_index_to_different_frame(input, inv_jacobian);
      static_assert(
          std::is_same_v<
              std::decay_t<decltype(result)>,
              Tensor<
                  double, Symmetry<1, 2>,
                  index_list<SpatialIndex<1, UpLo::Up, Frame::ElementLogical>,
                             SpatialIndex<1, UpLo::Lo, Frame::Inertial>>>>);
      CHECK(get<0, 0>(result) == 2.0);
    }
  }
  {
    INFO("2D");
    InverseJacobian<double, 2, Frame::ElementLogical, Frame::Inertial>
        inv_jacobian{};
    get<0, 0>(inv_jacobian) = 2.0;
    get<1, 1>(inv_jacobian) = 3.0;
    get<0, 1>(inv_jacobian) = 0.5;
    get<1, 0>(inv_jacobian) = 1.5;
    {
      INFO("Vector");
      const tnsr::I<double, 2, Frame::Inertial> input{{1.0, 2.0}};
      const auto result = first_index_to_different_frame(input, inv_jacobian);
      static_assert(std::is_same_v<std::decay_t<decltype(result)>,
                                   tnsr::I<double, 2, Frame::ElementLogical>>);
      CHECK(get<0>(result) == 3.);
      CHECK(get<1>(result) == 7.5);
    }
    {
      INFO("Rank 2 tensor");
      tnsr::Ij<double, 2, Frame::Inertial> input{};
      get<0, 0>(input) = 1.0;
      get<1, 0>(input) = 2.0;
      get<0, 1>(input) = 3.0;
      get<1, 1>(input) = 4.0;
      const auto result = first_index_to_different_frame(input, inv_jacobian);
      static_assert(
          std::is_same_v<
              std::decay_t<decltype(result)>,
              Tensor<
                  double, Symmetry<1, 2>,
                  index_list<SpatialIndex<2, UpLo::Up, Frame::ElementLogical>,
                             SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>>);
      CHECK(get<0, 0>(result) == 3.);
      CHECK(get<1, 0>(result) == 7.5);
      CHECK(get<0, 1>(result) == 8.);
      CHECK(get<1, 1>(result) == 16.5);
    }
  }
  {
    INFO("Variables");
    const size_t num_points = 3;
    InverseJacobian<DataVector, 2, Frame::ElementLogical, Frame::Inertial>
        inv_jacobian{num_points};
    get<0, 0>(inv_jacobian) = 2.0;
    get<1, 1>(inv_jacobian) = 3.0;
    get<0, 1>(inv_jacobian) = 0.5;
    get<1, 0>(inv_jacobian) = 1.5;
    Variables<tmpl::list<Var1, Var2>> input{num_points};
    std::iota(get<Var1>(input).begin(), get<Var1>(input).end(), 1.0);
    std::iota(get<Var2>(input).begin(), get<Var2>(input).end(), 1.0);
    CAPTURE(input);
    const auto result = first_index_to_different_frame(input, inv_jacobian);
    const auto& var1 =
        get<Tags::TransformedFirstIndex<Var1, Frame::ElementLogical>>(result);
    const auto& var2 =
        get<Tags::TransformedFirstIndex<Var2, Frame::ElementLogical>>(result);
    CHECK(get<0>(var1) == 3.);
    CHECK(get<1>(var1) == 7.5);
    CHECK(get<0, 0>(var2) == 3.);
    CHECK(get<1, 0>(var2) == 7.5);
    CHECK(get<0, 1>(var2) == 8.);
    CHECK(get<1, 1>(var2) == 16.5);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Tensor.EagerMath.FrameTransform",
                  "[DataStructures][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "DataStructures/Tensor/EagerMath/");
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
  test_transform_first_index_to_different_frame();
}

}  // namespace transform
