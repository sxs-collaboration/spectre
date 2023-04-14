// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"

#include <cmath>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/RotationMatrixHelpers.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::CoordinateMaps::TimeDependent {

template <size_t Dim>
Rotation<Dim>::Rotation(std::string function_of_time_name)
    : f_of_t_name_(std::move(function_of_time_name)) {}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> Rotation<Dim>::operator()(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const Matrix rot_matrix =
      rotation_matrix<Dim>(time, *(functions_of_time.at(f_of_t_name_)));

  std::array<tt::remove_cvref_wrap_t<T>, Dim> result{};
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = rot_matrix(i, 0) * source_coords[0];
    for (size_t j = 1; j < Dim; j++) {
      gsl::at(result, i) += rot_matrix(i, j) * gsl::at(source_coords, j);
    }
  }
  return result;
}

template <size_t Dim>
std::optional<std::array<double, Dim>> Rotation<Dim>::inverse(
    const std::array<double, Dim>& target_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const Matrix rot_matrix =
      rotation_matrix<Dim>(time, *(functions_of_time.at(f_of_t_name_)));

  // The inverse map uses the inverse rotation matrix, which is just the
  // transpose of the rotation matrix
  std::array<double, Dim> result{};
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = rot_matrix(0, i) * target_coords[0];
    for (size_t j = 1; j < Dim; j++) {
      gsl::at(result, i) += rot_matrix(j, i) * gsl::at(target_coords, j);
    }
  }
  return result;
}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> Rotation<Dim>::frame_velocity(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const Matrix rot_matrix_deriv =
      rotation_matrix_deriv<Dim>(time, *(functions_of_time.at(f_of_t_name_)));

  std::array<tt::remove_cvref_wrap_t<T>, Dim> result{};
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = rot_matrix_deriv(i, 0) * source_coords[0];
    for (size_t j = 1; j < Dim; j++) {
      gsl::at(result, i) += rot_matrix_deriv(i, j) * gsl::at(source_coords, j);
    }
  }

  return result;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
Rotation<Dim>::jacobian(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const Matrix rot_matrix =
      rotation_matrix<Dim>(time, *(functions_of_time.at(f_of_t_name_)));

  // Make tensor of zeros with correct type
  auto jacobian_matrix = make_with_value<
      tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);

  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      jacobian_matrix.get(i, j) = rot_matrix(i, j);
    }
  }

  return jacobian_matrix;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
Rotation<Dim>::inv_jacobian(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const Matrix rot_matrix =
      rotation_matrix<Dim>(time, *(functions_of_time.at(f_of_t_name_)));

  auto inv_jacobian_matrix = make_with_value<
      tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);

  // The inverse jacobian is just the inverse rotation matrix, which is the
  // transpose of the rotation matrix.
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      inv_jacobian_matrix.get(i, j) = rot_matrix(j, i);
    }
  }

  return inv_jacobian_matrix;
}

template <size_t Dim>
void Rotation<Dim>::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | f_of_t_name_;
  }
}

template <size_t Dim>
bool operator==(const Rotation<Dim>& lhs, const Rotation<Dim>& rhs) {
  return lhs.f_of_t_name_ == rhs.f_of_t_name_;
}

template <size_t Dim>
bool operator!=(const Rotation<Dim>& lhs, const Rotation<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template class Rotation<DIM(data)>;                                      \
  template bool operator==                                                 \
      <DIM(data)>(const Rotation<DIM(data)>&, const Rotation<DIM(data)>&); \
  template bool operator!=                                                 \
      <DIM(data)>(const Rotation<DIM(data)>&, const Rotation<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef INSTANTIATE
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>      \
  Rotation<DIM(data)>::operator()(                                          \
      const std::array<DTYPE(data), DIM(data)>& source_coords, double time, \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const;                                         \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>      \
  Rotation<DIM(data)>::frame_velocity(                                      \
      const std::array<DTYPE(data), DIM(data)>& source_coords,              \
      const double time,                                                    \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const;                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),        \
                    Frame::NoFrame>                                         \
  Rotation<DIM(data)>::jacobian(                                            \
      const std::array<DTYPE(data), DIM(data)>& source_coords, double time, \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const;                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),        \
                    Frame::NoFrame>                                         \
  Rotation<DIM(data)>::inv_jacobian(                                        \
      const std::array<DTYPE(data), DIM(data)>& source_coords, double time, \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef DIM
#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps::TimeDependent
