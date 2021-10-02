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
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
Matrix rotation_matrix(
    const std::string& f_of_t_name, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  ASSERT(functions_of_time.count(f_of_t_name) == 1,
         "The function of time '" << f_of_t_name
                                  << "' is not one of the known functions of "
                                     "time. The known functions of time are: "
                                  << keys_of(functions_of_time));
  const double rotation_angle =
      functions_of_time.at(f_of_t_name)->func(time)[0][0];
  return Matrix{{cos(rotation_angle), -sin(rotation_angle)},
                          {sin(rotation_angle), cos(rotation_angle)}};
}
}  // namespace

namespace domain::CoordinateMaps::TimeDependent {

Rotation<2>::Rotation(std::string function_of_time_name)
    : f_of_t_name_(std::move(function_of_time_name)) {}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 2> Rotation<2>::operator()(
    const std::array<T, 2>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const Matrix rot_matrix =
      rotation_matrix(f_of_t_name_, time, functions_of_time);
  return {{source_coords[0] * rot_matrix(0, 0) +
               source_coords[1] * rot_matrix(0, 1),
           source_coords[0] * rot_matrix(1, 0) +
               source_coords[1] * rot_matrix(1, 1)}};
}

std::optional<std::array<double, 2>> Rotation<2>::inverse(
    const std::array<double, 2>& target_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const Matrix rot_matrix =
      rotation_matrix(f_of_t_name_, time, functions_of_time);
  // The inverse map uses the inverse rotation matrix, which is just the
  // transpose of the rotation matrix
  return {{{target_coords[0] * rot_matrix(0, 0) +
                target_coords[1] * rot_matrix(1, 0),
            target_coords[0] * rot_matrix(0, 1) +
                target_coords[1] * rot_matrix(1, 1)}}};
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 2> Rotation<2>::frame_velocity(
    const std::array<T, 2>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  // The mapped coordinates (x,y) are related to the unmapped
  // coordinates (\xi, \eta) by
  //   x = \cos(\alpha) \xi - \sin(\alpha) \eta
  //   y = \sin(\alpha) \xi + \cos(\alpha) \eta
  //
  // The frame velocity is
  //   dx/dt = (-\sin(\alpha) \xi - \cos(\alpha)\eta) * d\alpha/dt
  //   dy/dt = (\cos(\alpha) \xi -\sin(\alpha) \eta) * d\alpha/dt
  const Matrix rot_matrix =
      rotation_matrix(f_of_t_name_, time, functions_of_time);
  const double rotation_angular_velocity =
      functions_of_time.at(f_of_t_name_)->func_and_deriv(time)[1][0];
  return {{(source_coords[0] * rot_matrix(0, 1) -
            source_coords[1] * rot_matrix(0, 0)) *
               rotation_angular_velocity,
           (source_coords[0] * rot_matrix(0, 0) +
            source_coords[1] * rot_matrix(0, 1)) *
               rotation_angular_velocity}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> Rotation<2>::jacobian(
    const std::array<T, 2>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const Matrix rot_matrix =
      rotation_matrix(f_of_t_name_, time, functions_of_time);
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> jacobian_matrix{
      make_with_value<tt::remove_cvref_wrap_t<T>>(
          dereference_wrapper(source_coords[0]), rot_matrix(0, 0))};
  // rot_matrix(0, 0) == rot_matrix(1, 1), so only set off-diagonal terms
  get<1, 0>(jacobian_matrix) = rot_matrix(1, 0);
  get<0, 1>(jacobian_matrix) = rot_matrix(0, 1);
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame>
Rotation<2>::inv_jacobian(
    const std::array<T, 2>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const Matrix rot_matrix =
      rotation_matrix(f_of_t_name_, time, functions_of_time);
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> inv_jacobian_matrix{
      make_with_value<tt::remove_cvref_wrap_t<T>>(
          dereference_wrapper(source_coords[0]), rot_matrix(0, 0))};
  // The inverse jacobian is just the inverse rotation matrix, which is the
  // transpose of the rotation matrix.
  // Also, rot_matrix(0, 0) == rot_matrix(1, 1), so only set off-diagonal terms
  get<1, 0>(inv_jacobian_matrix) = rot_matrix(0, 1);
  get<0, 1>(inv_jacobian_matrix) = rot_matrix(1, 0);
  return inv_jacobian_matrix;
}

void Rotation<2>::pup(PUP::er& p) { p | f_of_t_name_; }

bool operator==(const Rotation<2>& lhs, const Rotation<2>& rhs) {
  return lhs.f_of_t_name_ == rhs.f_of_t_name_;
}

bool operator!=(const Rotation<2>& lhs, const Rotation<2>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
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

GENERATE_INSTANTIATIONS(INSTANTIATE, (2),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))
#undef DIM
#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps::TimeDependent
