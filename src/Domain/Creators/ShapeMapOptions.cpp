// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/ShapeMapOptions.hpp"

#include <array>
#include <string>
#include <utility>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain::creators::time_dependent_options {
template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
std::pair<std::array<DataVector, 3>, std::array<DataVector, 4>>
initial_shape_and_size_funcs(
    const ShapeMapOptions<IncludeTransitionEndsAtCube, Object>& shape_options,
    const double inner_radius) {
  const DataVector shape_zeros{
      ylm::Spherepack::spectral_size(shape_options.l_max, shape_options.l_max),
      0.0};

  std::array<DataVector, 3> shape_funcs =
      make_array<3, DataVector>(shape_zeros);
  std::array<DataVector, 4> size_funcs =
      make_array<4, DataVector>(DataVector{1, 0.0});

  if (shape_options.initial_values.has_value()) {
    if (std::holds_alternative<KerrSchildFromBoyerLindquist>(
            shape_options.initial_values.value())) {
      const ylm::Spherepack ylm{shape_options.l_max, shape_options.l_max};
      const auto& mass_and_spin = std::get<KerrSchildFromBoyerLindquist>(
          shape_options.initial_values.value());
      const DataVector radial_distortion =
          inner_radius -
          get(gr::Solutions::kerr_schild_radius_from_boyer_lindquist(
              inner_radius, ylm.theta_phi_points(), mass_and_spin.mass,
              mass_and_spin.spin));
      shape_funcs[0] = ylm.phys_to_spec(radial_distortion);
      // Transform from SPHEREPACK to actual Ylm for size func
      size_funcs[0][0] = shape_funcs[0][0] * sqrt(0.5 * M_PI);
      // Set l=0 for shape map to 0 because size is going to be used
      shape_funcs[0][0] = 0.0;
    }
  }

  // If any size options were specified, those override the values from the
  // shape coefs
  if (shape_options.initial_size_values.has_value()) {
    for (size_t i = 0; i < 3; i++) {
      gsl::at(size_funcs, i)[0] =
          gsl::at(shape_options.initial_size_values.value(), i);
    }
  }

  return std::make_pair(std::move(shape_funcs), std::move(size_funcs));
}

#define INCLUDETRANSITION(data) BOOST_PP_TUPLE_ELEM(0, data)
#define OBJECT(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                               \
  template class ShapeMapOptions<INCLUDETRANSITION(data), OBJECT(data)>;   \
  template std::pair<std::array<DataVector, 3>, std::array<DataVector, 4>> \
  initial_shape_and_size_funcs<INCLUDETRANSITION(data), OBJECT(data)>(     \
      const ShapeMapOptions<INCLUDETRANSITION(data), OBJECT(data)>&        \
          shape_options,                                                   \
      double inner_radius);

GENERATE_INSTANTIATIONS(INSTANTIATE, (true, false),
                        (domain::ObjectLabel::A, domain::ObjectLabel::B,
                         domain::ObjectLabel::None))

#undef INCLUDETRANSITION
#undef OBJECT
#undef INSTANTIATE

}  // namespace domain::creators::time_dependent_options
