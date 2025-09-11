// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Python/Composition.hpp"

#include <pybind11/pybind11.h>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Composition.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/SnakeCase.hpp"
#include "Utilities/TMPL.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {

template <typename Frames, size_t Dim>
void bind_composition_impl(py::module& m) {  // NOLINT
  using CompositionType = CoordinateMaps::Composition<Frames, Dim>;
  std::string frames_string{};
  tmpl::for_each<Frames>(
      [&frames_string]<typename Frame>(const tmpl::type_<Frame> /*meta*/) {
        frames_string += get_output(Frame{});
      });
  auto binding = py::class_<CompositionType, typename CompositionType::Base>(
      m, ("CompositionMap" + frames_string + get_output(Dim) + "D").c_str());
  tmpl::for_each<
      tmpl::pop_back<Frames>>([&binding]<typename SourceFrame>(
                                  const tmpl::type_<SourceFrame> /*meta*/) {
    using TargetFrame =
        tmpl::at<Frames,
                 tmpl::size_t<tmpl::index_of<Frames, SourceFrame>::value + 1>>;
    binding.def_property_readonly(
        (camel_case_to_snake_case(get_output(SourceFrame{})) + "_to_" +
         camel_case_to_snake_case(get_output(TargetFrame{})))
            .c_str(),
        &CompositionType::template get_component<SourceFrame, TargetFrame>,
        py::return_value_policy::reference_internal);
  });
}
}  // namespace

void bind_composition(py::module& m) {  // NOLINT
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  bind_composition_impl<                                                       \
      tmpl::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Grid,      \
                 Frame::Distorted, Frame::Inertial>,                           \
      DIM(data)>(m);                                                           \
  bind_composition_impl<tmpl::list<Frame::ElementLogical, Frame::BlockLogical, \
                                   Frame::Grid, Frame::Inertial>,              \
                        DIM(data)>(m);                                         \
  bind_composition_impl<                                                       \
      tmpl::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Inertial>, \
      DIM(data)>(m);

  GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}
}  // namespace domain::py_bindings
