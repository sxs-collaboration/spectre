// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"

#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::CoordinateMaps::TimeDependent {

template <size_t Dim>
Translation<Dim>::Translation(std::string function_of_time_name) noexcept
    : f_of_t_name_(std::move(function_of_time_name)) {}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> Translation<Dim>::operator()(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.count(f_of_t_name_) == 1,
         "The function of time '" << f_of_t_name_
                                  << "' is not one of the known functions of "
                                     "time. The known functions of time are: "
                                  << keys_of(functions_of_time));
  std::array<tt::remove_cvref_wrap_t<T>, Dim> result{};
  const DataVector function_of_time =
      functions_of_time.at(f_of_t_name_)->func(time)[0];
  ASSERT(function_of_time.size() == Dim,
         "The dimension of the function of time ("
             << function_of_time.size()
             << ") does not match the dimension of the translation map (" << Dim
             << ").");
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(source_coords, i) + function_of_time[i];
  }
  return result;
}

template <size_t Dim>
std::optional<std::array<double, Dim>> Translation<Dim>::inverse(
    const std::array<double, Dim>& target_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.count(f_of_t_name_) == 1,
         "The function of time '" << f_of_t_name_
                                  << "' is not one of the known functions of "
                                     "time. The known functions of time are: "
                                  << keys_of(functions_of_time));
  std::array<double, Dim> result{};
  const DataVector function_of_time =
      functions_of_time.at(f_of_t_name_)->func(time)[0];
  ASSERT(function_of_time.size() == Dim,
         "The dimension of the function of time ("
             << function_of_time.size()
             << ") does not match the dimension of the translation map (" << Dim
             << ").");
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(target_coords, i) - function_of_time[i];
  }
  return result;
}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> Translation<Dim>::frame_velocity(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.count(f_of_t_name_) == 1,
         "The function of time '" << f_of_t_name_
                                  << "' is not one of the known functions of "
                                     "time. The known functions of time are: "
                                  << keys_of(functions_of_time));
  std::array<tt::remove_cvref_wrap_t<T>, Dim> result{};
  const auto function_of_time_and_deriv =
      functions_of_time.at(f_of_t_name_)->func_and_deriv(time);
  const DataVector& velocity = function_of_time_and_deriv[1];
  ASSERT(velocity.size() == Dim,
         "The dimension of the function of time ("
             << velocity.size()
             << ") does not match the dimension of the translation map (" << Dim
             << ").");
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = make_with_value<tt::remove_cvref_wrap_t<T>>(
        dereference_wrapper(gsl::at(source_coords, i)), velocity[i]);
  }
  return result;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
Translation<Dim>::jacobian(
    const std::array<T, Dim>& source_coords) const noexcept {
  return identity<Dim>(dereference_wrapper(source_coords[0]));
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
Translation<Dim>::inv_jacobian(
    const std::array<T, Dim>& source_coords) const noexcept {
  return identity<Dim>(dereference_wrapper(source_coords[0]));
}

template <size_t Dim>
void Translation<Dim>::pup(PUP::er& p) noexcept {
  p | f_of_t_name_;
}

template <size_t Dim>
bool operator==(const Translation<Dim>& lhs,
                const Translation<Dim>& rhs) noexcept {
  return lhs.f_of_t_name_ == rhs.f_of_t_name_;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                              \
  template class Translation<DIM(data)>;                  \
  template bool operator==(const Translation<DIM(data)>&, \
                           const Translation<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>         \
  Translation<DIM(data)>::operator()(                                          \
      const std::array<DTYPE(data), DIM(data)>& source_coords,                 \
      const double time,                                                       \
      const std::unordered_map<                                                \
          std::string,                                                         \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&           \
          functions_of_time) const noexcept;                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>         \
  Translation<DIM(data)>::frame_velocity(                                      \
      const std::array<DTYPE(data), DIM(data)>& source_coords,                 \
      const double time,                                                       \
      const std::unordered_map<                                                \
          std::string,                                                         \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&           \
          functions_of_time) const noexcept;                                   \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),           \
                    Frame::NoFrame>                                            \
  Translation<DIM(data)>::jacobian(                                            \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),           \
                    Frame::NoFrame>                                            \
  Translation<DIM(data)>::inv_jacobian(                                        \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))
#undef DIM
#undef DTYPE
#undef INSTANTIATE
}  // namespace domain::CoordinateMaps::TimeDependent
