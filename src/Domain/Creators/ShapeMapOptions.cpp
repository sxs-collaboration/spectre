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
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
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
    } else if (std::holds_alternative<YlmsFromFile>(
                   shape_options.initial_values.value())) {
      const auto& files =
          std::get<YlmsFromFile>(shape_options.initial_values.value());
      const std::string& h5_filename = files.h5_filename;
      const std::vector<std::string>& subfile_names = files.subfile_names;
      const double match_time = files.match_time;
      const double match_time_epsilon =
          files.match_time_epsilon.value_or(1e-12);
      const bool set_l1_coefs_to_zero = files.set_l1_coefs_to_zero;
      const size_t l_max = shape_options.l_max;
      ylm::SpherepackIterator iter{l_max, l_max};

      for (size_t i = 0; i < subfile_names.size(); i++) {
        // Frame doesn't matter here
        const ylm::Strahlkorper<Frame::Distorted> file_strahlkorper =
            ylm::read_surface_ylm_single_time<Frame::Distorted>(
                h5_filename, gsl::at(subfile_names, i), match_time,
                match_time_epsilon, files.check_frame);
        const ylm::Strahlkorper<Frame::Distorted> this_strahlkorper{
            shape_options.l_max, 1.0, std::array{0.0, 0.0, 0.0}};

        // The coefficients in the shape map are stored as the negative
        // coefficients of the strahlkorper, so we need to multiply by -1 here.
        gsl::at(shape_funcs, i) =
            -1.0 * file_strahlkorper.ylm_spherepack().prolong_or_restrict(
                       file_strahlkorper.coefficients(),
                       this_strahlkorper.ylm_spherepack());
        // Transform from SPHEREPACK to actual Ylm for size func
        gsl::at(size_funcs, i)[0] =
            gsl::at(shape_funcs, i)[0] * sqrt(0.5 * M_PI);
        // Set l=0 for shape map to 0 because size is going to be used
        gsl::at(shape_funcs, i)[0] = 0.0;
        if (set_l1_coefs_to_zero) {
          for (int m = -1; m <= 1; m++) {
            gsl::at(shape_funcs, i)[iter.set(1_st, m)()] = 0.0;
          }
        }
      }
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
