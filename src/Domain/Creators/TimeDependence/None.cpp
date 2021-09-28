// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/None.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::creators::time_dependence {
template <size_t MeshDim>
std::unique_ptr<TimeDependence<MeshDim>> None<MeshDim>::get_clone() const {
  return std::make_unique<None>(*this);
}

template <size_t MeshDim>
[[noreturn]] std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
None<MeshDim>::block_maps(const size_t /*number_of_blocks*/) const {
  ERROR(
      "The 'block_maps' function of the 'None' TimeDependence should never be "
      "called because 'None' is only used as a place holder class to mark that "
      "the mesh is time-independent.");
}

template <size_t MeshDim>
[[noreturn]] std::unordered_map<
    std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
None<MeshDim>::functions_of_time() const {
  ERROR(
      "The 'functions_of_time' function of the 'None' TimeDependence should "
      "never be  called because 'None' is only used as a place holder class to "
      "mark that the mesh is time-independent.");
}

template <size_t Dim>
bool operator==(const None<Dim>& /*lhs*/, const None<Dim>& /*rhs*/) {
  return true;
}

template <size_t Dim>
bool operator!=(const None<Dim>& lhs, const None<Dim>& rhs) {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template class None<GET_DIM(data)>;                                          \
  template bool operator==                                                     \
      <GET_DIM(data)>(const None<GET_DIM(data)>&, const None<GET_DIM(data)>&); \
  template bool operator!=                                                     \
      <GET_DIM(data)>(const None<GET_DIM(data)>&, const None<GET_DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace domain::creators::time_dependence
