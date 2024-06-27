// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "FromVolumeFile.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain::creators::time_dependent_options {
KerrSchildFromBoyerLindquist::KerrSchildFromBoyerLindquist() = default;
KerrSchildFromBoyerLindquist::KerrSchildFromBoyerLindquist(
    const double mass_in, const std::array<double, 3> spin_in)
    : mass(mass_in), spin(spin_in) {}

YlmsFromFile::YlmsFromFile() = default;
YlmsFromFile::YlmsFromFile(std::string h5_filename_in,
                           std::vector<std::string> subfile_names_in,
                           double match_time_in,
                           std::optional<double> match_time_epsilon_in,
                           bool set_l1_coefs_to_zero_in, bool check_frame_in)
    : h5_filename(std::move(h5_filename_in)),
      subfile_names(std::move(subfile_names_in)),
      match_time(match_time_in),
      match_time_epsilon(match_time_epsilon_in),
      set_l1_coefs_to_zero(set_l1_coefs_to_zero_in),
      check_frame(check_frame_in) {}

YlmsFromSpEC::YlmsFromSpEC() = default;
YlmsFromSpEC::YlmsFromSpEC(std::string dat_filename_in,
                           const double match_time_in,
                           const std::optional<double> match_time_epsilon_in,
                           bool set_l1_coefs_to_zero_in)
    : dat_filename(std::move(dat_filename_in)),
      match_time(match_time_in),
      match_time_epsilon(match_time_epsilon_in),
      set_l1_coefs_to_zero(set_l1_coefs_to_zero_in) {}

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
      // Set l=0 for shape map to 0 because size control will adjust l=0
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
        // Set l=0 for shape map to 0 because size control will adjust l=0
        gsl::at(shape_funcs, i)[0] = 0.0;
        if (set_l1_coefs_to_zero) {
          for (int m = -1; m <= 1; m++) {
            gsl::at(shape_funcs, i)[iter.set(1_st, m)()] = 0.0;
          }
        }
      }
    } else if (std::holds_alternative<YlmsFromSpEC>(
                   shape_options.initial_values.value())) {
      const auto& spec_option =
          std::get<YlmsFromSpEC>(shape_options.initial_values.value());
      const std::string& dat_filename = spec_option.dat_filename;
      const double match_time = spec_option.match_time;
      const double match_time_epsilon =
          spec_option.match_time_epsilon.value_or(1e-12);
      const bool set_l1_coefs_to_zero = spec_option.set_l1_coefs_to_zero;

      std::ifstream dat_file(dat_filename);
      if (not dat_file.is_open()) {
        ERROR("Unable to open SpEC dat file " << dat_filename);
      }
      std::string line{};
      size_t total_col = 0;
      std::optional<size_t> l_max{};
      std::array<double, 3> center{};
      ModalVector coefficients{};
      // This will be actually set below
      ylm::SpherepackIterator file_iter{2, 2};

      // We have to parse the dat file manually
      while (std::getline(dat_file, line)) {
        // Avoid comment lines. The SpEC file puts the legend in comments at the
        // top of the file, so we count how many columns the dat file has based
        // on the number of comment lines that are the legend (ends in ')')
        if (line.starts_with("#")) {
          if (line.starts_with("# [") and line.ends_with(")")) {
            ++total_col;
          }
          continue;
        }

        std::stringstream ss(line);

        double time = 0.0;
        ss >> time;

        // Set scale to current time plus 1 just in case time = 0
        if (not equal_within_roundoff(time, match_time, match_time_epsilon,
                                      time + 1.0)) {
          continue;
        }

        if (l_max.has_value()) {
          ERROR("Found more than one time in the SpEC dat file "
                << dat_filename << " that is within a relative epsilon of "
                << match_time_epsilon << " of the time requested " << time);
        }

        // Casting to an integer floors a double, so we add 0.5 before we take
        // the sqrt to avoid any rounding issues
        const auto l_max_plus_one =
            static_cast<size_t>(sqrt(static_cast<double>(total_col) + 0.5));
        if (l_max_plus_one == 0) {
          ERROR(
              "Invalid l_max from SpEC dat file. l_max + 1 was computed to be "
              "0");
        }
        l_max = l_max_plus_one - 1;

        ss >> center[0];
        ss >> center[1];
        ss >> center[2];

        coefficients.destructive_resize(
            ylm::Spherepack::spectral_size(l_max.value(), l_max.value()));

        file_iter = ylm::SpherepackIterator{l_max.value(), l_max.value()};

        for (int l = 0; l <= static_cast<int>(l_max.value()); l++) {
          for (int m = -l; m <= l; m++) {
            ss >> coefficients[file_iter.set(static_cast<size_t>(l), m)()];
          }
        }
      }

      if (not l_max.has_value()) {
        ERROR_NO_TRACE("Unable to find requested time "
                       << time << " within an epsilon of " << match_time_epsilon
                       << " in SpEC dat file " << dat_filename);
      }

      const ylm::Strahlkorper<Frame::Inertial> file_strahlkorper{
          l_max.value(), l_max.value(), coefficients, center};
      const ylm::Strahlkorper<Frame::Inertial> this_strahlkorper{
          shape_options.l_max, 1.0, std::array{0.0, 0.0, 0.0}};
      ylm::SpherepackIterator iter{shape_options.l_max, shape_options.l_max};

      shape_funcs[0] =
          -1.0 * file_strahlkorper.ylm_spherepack().prolong_or_restrict(
                     file_strahlkorper.coefficients(),
                     this_strahlkorper.ylm_spherepack());
      // Transform from SPHEREPACK to actual Ylm for size func
      size_funcs[0][0] = shape_funcs[0][0] * sqrt(0.5 * M_PI);
      // Set l=0 for shape map to 0 because size control will adjust l=0
      shape_funcs[0][0] = 0.0;
      if (set_l1_coefs_to_zero) {
        for (int m = -1; m <= 1; m++) {
          shape_funcs[0][iter.set(1_st, m)()] = 0.0;
        }
      }
    } else if (std::holds_alternative<FromVolumeFile<names::ShapeSize<Object>>>(
                   shape_options.initial_values.value())) {
      const auto& volume_file_options =
          std::get<FromVolumeFile<names::ShapeSize<Object>>>(
              shape_options.initial_values.value());

      shape_funcs = volume_file_options.shape_values;
      size_funcs = volume_file_options.size_values;
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
