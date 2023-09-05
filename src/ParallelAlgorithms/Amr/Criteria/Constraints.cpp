// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/Constraints.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace amr::Criteria::Constraints_detail {

template <size_t Dim>
void normalization_factor_square(
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::ElementLogical>*>
        result,
    const Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>&
        jacobian) {
  set_number_of_grid_points(result, jacobian);
  for (size_t k = 0; k < Dim; ++k) {
    // Possible performance optimization: unroll first iteration of loop to
    // avoid zeroing buffer
    result->get(k) = 0.;
    for (size_t i = 0; i < Dim; ++i) {
      result->get(k) += square(jacobian.get(i, k));
    }
  }
}

template <size_t Dim, typename TensorType>
void logical_constraints(
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::ElementLogical>*>
        result,
    const gsl::not_null<DataVector*> buffer,
    const TensorType& constraints_tensor,
    const Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>&
        jacobian,
    const tnsr::i<DataVector, Dim, Frame::ElementLogical>&
        normalization_factor_square) {
  static_assert(TensorType::rank() >= 2,
                "The constraints tensor must have rank 2 or higher.");
  using first_index = tmpl::front<typename TensorType::index_list>;
  static_assert(
      first_index::index_type == IndexType::Spatial and
          first_index::dim == Dim and first_index::ul == UpLo::Lo and
          std::is_same_v<typename first_index::Frame, Frame::Inertial>,
      "The first index of the constraints tensor must be a lower "
      "spatial index (that originates from a derivative).");
  using NonFirstIndexTensor =
      TensorMetafunctions::remove_first_index<TensorType>;
  set_number_of_grid_points(result, constraints_tensor);
  set_number_of_grid_points(buffer, constraints_tensor);
  for (size_t k = 0; k < Dim; ++k) {
    result->get(k) = 0.;
    for (size_t a = 0; a < NonFirstIndexTensor::size(); ++a) {
      const auto non_first_index = NonFirstIndexTensor::get_tensor_index(a);
      // Possible performance optimization: unroll first iteration of loop to
      // avoid zeroing buffer
      *buffer = 0.;
      for (size_t i = 0; i < Dim; ++i) {
        const auto full_index = prepend(non_first_index, i);
        *buffer += jacobian.get(i, k) * constraints_tensor.get(full_index);
      }
      result->get(k) += square(*buffer);
    }
    result->get(k) = sqrt(result->get(k) / normalization_factor_square.get(k));
  }
}

template <size_t Dim>
void max_over_components(
    const gsl::not_null<std::array<Flag, Dim>*> result,
    const tnsr::i<DataVector, Dim, Frame::ElementLogical>& logical_constraints,
    const double abs_target, const double coarsening_factor) {
  // We take the highest-priority refinement flag in each dimension, so if any
  // constraint is above the target, the element will increase p refinement in
  // that dimension. And only if all constraints are below the coarsening
  // threshold will the element decrease p refinement in that dimension.
  const size_t sqrt_num_points = sqrt(logical_constraints.begin()->size());
  for (size_t d = 0; d < Dim; ++d) {
    // Skip this dimension if we have already decided to refine it
    if (gsl::at(*result, d) == Flag::IncreaseResolution) {
      continue;
    }
    const double constraint_norm =
        blaze::l2Norm(logical_constraints.get(d)) / sqrt_num_points;
    // Increase p refinement if the constraint norm exceeds the target
    if (constraint_norm > abs_target) {
      gsl::at(*result, d) = Flag::IncreaseResolution;
      continue;
    }
    // Dont' check if we want to (allow) decreasing p refinement if another
    // tensor has already decided that decreasing p refinement is bad.
    if (gsl::at(*result, d) == Flag::DoNothing) {
      continue;
    }
    // Decrease p refinement if the constraint norm is below the coarsening
    // threshold. Otherwise, request to stay at this resolution (or increase
    // resolution if another constraint requested that).
    if (constraint_norm < coarsening_factor * abs_target) {
      gsl::at(*result, d) = Flag::DecreaseResolution;
    } else {
      gsl::at(*result, d) = Flag::DoNothing;
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TNSR(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                       \
  template void normalization_factor_square(                       \
      const gsl::not_null<                                         \
          tnsr::i<DataVector, DIM(data), Frame::ElementLogical>*>  \
          result,                                                  \
      const Jacobian<DataVector, DIM(data), Frame::ElementLogical, \
                     Frame::Inertial>& jacobian);                  \
  template void max_over_components(                               \
      gsl::not_null<std::array<Flag, DIM(data)>*> result,          \
      const tnsr::i<DataVector, DIM(data), Frame::ElementLogical>& \
          logical_constraints,                                     \
      double abs_target, double coarsening_factor);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#define INSTANTIATE_TENSORS(_, data)                                          \
  template void logical_constraints(                                          \
      gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::ElementLogical>*>   \
          result,                                                             \
      gsl::not_null<DataVector*> buffer, const tnsr::TNSR(data) < DataVector, \
      DIM(data), Frame::Inertial > &constraints_tensor,                       \
      const Jacobian<DataVector, DIM(data), Frame::ElementLogical,            \
                     Frame::Inertial>& jacobian,                              \
      const tnsr::i<DataVector, DIM(data), Frame::ElementLogical>&            \
          normalization_factor_square);

GENERATE_INSTANTIATIONS(INSTANTIATE_TENSORS, (1, 2, 3), (ia, iaa))

#undef INSTANTIATE
#undef INSTANTIATE_TENSORS
#undef DIM

}  // namespace amr::Criteria::Constraints_detail
