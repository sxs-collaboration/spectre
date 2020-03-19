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
#include "ErrorHandling/Assert.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain {
namespace CoordMapsTimeDependent {

Translation::Translation(std::string function_of_time_name) noexcept
    : f_of_t_name_(std::move(function_of_time_name)) {}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> Translation::operator()(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.count(f_of_t_name_) == 1,
         "The function of time '" << f_of_t_name_
                                  << "' is not one of the known functions of "
                                     "time. The known functions of time are: "
                                  << keys_of(functions_of_time));
  return {{source_coords[0] +
           functions_of_time.at(f_of_t_name_)->func(time)[0][0]}};
}

boost::optional<std::array<double, 1>> Translation::inverse(
    const std::array<double, 1>& target_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.count(f_of_t_name_) == 1,
         "The function of time '" << f_of_t_name_
                                  << "' is not one of the known functions of "
                                     "time. The known functions of time are: "
                                  << keys_of(functions_of_time));
  return {{{target_coords[0] -
            functions_of_time.at(f_of_t_name_)->func(time)[0][0]}}};
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> Translation::frame_velocity(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.count(f_of_t_name_) == 1,
         "The function of time '" << f_of_t_name_
                                  << "' is not one of the known functions of "
                                     "time. The known functions of time are: "
                                  << keys_of(functions_of_time));
  return {{make_with_value<tt::remove_cvref_wrap_t<T>>(
      dereference_wrapper(source_coords[0]),
      functions_of_time.at(f_of_t_name_)->func_and_deriv(time)[1][0])}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> Translation::jacobian(
    const std::array<T, 1>& source_coords) const noexcept {
  return identity<1>(dereference_wrapper(source_coords[0]));
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>
Translation::inv_jacobian(const std::array<T, 1>& source_coords) const
    noexcept {
  return identity<1>(dereference_wrapper(source_coords[0]));
}

void Translation::pup(PUP::er& p) noexcept { p | f_of_t_name_; }

bool operator==(const CoordMapsTimeDependent::Translation& lhs,
                const CoordMapsTimeDependent::Translation& rhs) noexcept {
  return lhs.f_of_t_name_ == rhs.f_of_t_name_;
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 1>               \
  Translation::operator()(                                                   \
      const std::array<DTYPE(data), 1>& source_coords, const double time,    \
      const std::unordered_map<                                              \
          std::string,                                                       \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&         \
          functions_of_time) const noexcept;                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 1>               \
  Translation::frame_velocity(                                               \
      const std::array<DTYPE(data), 1>& source_coords, const double time,    \
      const std::unordered_map<                                              \
          std::string,                                                       \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&         \
          functions_of_time) const noexcept;                                 \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame> \
  Translation::jacobian(const std::array<DTYPE(data), 1>& source_coords)     \
      const noexcept;                                                        \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame> \
  Translation::inv_jacobian(const std::array<DTYPE(data), 1>& source_coords) \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordMapsTimeDependent
}  // namespace domain
