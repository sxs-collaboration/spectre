// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <istream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "FromVolumeFile.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
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

template <ObjectLabel Object>
FromVolumeFileShapeSize<Object>::FromVolumeFileShapeSize(
    const std::optional<size_t>& l_max_in,
    const bool transition_ends_at_cube_in, std::string h5_filename,
    std::string subfile_name, const Options::Context& context)
    : FromVolumeFile(std::move(h5_filename), std::move(subfile_name)),
      transition_ends_at_cube(transition_ends_at_cube_in) {
  if (this->replay() and l_max_in.has_value()) {
    PARSE_ERROR(context,
                "Cannot supply LMax for shape map while also replaying.");
  }

  if (l_max_in.has_value()) {
    l_max = l_max_in.value();
  } else {
    const auto shape_fot_map = this->retrieve_function_of_time(
        {std::string{"Shape" + name(Object)}}, std::nullopt);
    const auto& shape_fot = shape_fot_map.at("Shape" + name(Object));

    const double initial_time = shape_fot->time_bounds()[0];
    const auto function = shape_fot->func(initial_time);

    // num_components = 2 * (l_max + 1)**2 if l_max == m_max which it is for the
    // shape map. This is why we can divide by 2 and take the sqrt without
    // worrying about odd numbers or non-perfect squares
    l_max = -1 + sqrt(function[0].size() / 2);  // NOLINT
  }
}

template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
size_t l_max_from_shape_options(
    const std::variant<ShapeMapOptions<IncludeTransitionEndsAtCube, Object>,
                       FromVolumeFileShapeSize<Object>>& shape_map_options) {
  return std::visit([](auto variant) { return variant.l_max; },
                    shape_map_options);
}

template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
bool transition_ends_at_cube_from_shape_options(
    const std::variant<ShapeMapOptions<IncludeTransitionEndsAtCube, Object>,
                       FromVolumeFileShapeSize<Object>>& shape_map_options) {
  return std::visit(
      [](auto variant) { return variant.transition_ends_at_cube; },
      shape_map_options);
}

template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
FunctionsOfTimeMap get_shape_and_size(
    const std::variant<ShapeMapOptions<IncludeTransitionEndsAtCube, Object>,
                       FromVolumeFileShapeSize<Object>>& shape_map_options,
    const double initial_time, const double shape_expiration_time,
    const double size_expiration_time, const double deformed_radius) {
  const size_t l_max = l_max_from_shape_options(shape_map_options);
  const DataVector shape_zeros{ylm::Spherepack::spectral_size(l_max, l_max),
                               0.0};
  const std::string shape_name = "Shape" + name(Object);
  const std::string size_name = "Size" + name(Object);

  FunctionsOfTimeMap result{};

  if (std::holds_alternative<FromVolumeFileShapeSize<Object>>(
          shape_map_options)) {
    const auto& from_vol_file =
        std::get<FromVolumeFileShapeSize<Object>>(shape_map_options);
    auto volume_fots = from_vol_file.retrieve_function_of_time(
        {shape_name, size_name}, initial_time);

    const auto check_fot = [&]<size_t MaxDeriv>(const std::string& name) {
      // It must be a PiecewisePolynomial
      if (UNLIKELY(dynamic_cast<
                       domain::FunctionsOfTime::PiecewisePolynomial<MaxDeriv>*>(
                       volume_fots.at(name).get()) == nullptr)) {
        ERROR_NO_TRACE(name << " function of time read from volume data is not "
                               "a PiecewisePolynomial<"
                            << MaxDeriv << ">. Cannot use it to initialize the "
                            << name << " map.");
      }
    };

    check_fot.template operator()<2>(shape_name);
    check_fot.template operator()<3>(size_name);

    if (from_vol_file.replay()) {
      result[shape_name] = std::move(volume_fots.at(shape_name));
    } else {
      // Not const in case we move it below
      auto temporary_shape_fot =
          volume_fots.at(shape_name)
              ->create_at_time(initial_time, shape_expiration_time);
      const auto shape_funcs =
          temporary_shape_fot->func_and_2_derivs(initial_time);

      const size_t file_l_max = -1 + sqrt(shape_funcs[0].size() / 2);  // NOLINT

      // Prolong or restrict if necessary, otherwise just use the exact function
      // of time from the volume file
      if (file_l_max != l_max) {
        std::array<DataVector, 3> new_shape_funcs{};
        const ylm::Spherepack file_spherepack{file_l_max, file_l_max};
        const ylm::Spherepack new_spherepack{l_max, l_max};
        for (size_t i = 0; i < shape_funcs.size(); i++) {
          gsl::at(new_shape_funcs, i) = file_spherepack.prolong_or_restrict(
              gsl::at(shape_funcs, i), new_spherepack);
        }

        result[shape_name] =
            std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
                initial_time, std::move(new_shape_funcs),
                shape_expiration_time);
      } else {
        result[shape_name] = std::move(temporary_shape_fot);
      }
    }

    if (from_vol_file.replay()) {
      result[size_name] = std::move(volume_fots.at(size_name));
    } else {
      result[size_name] = volume_fots.at(size_name)->create_at_time(
          initial_time, size_expiration_time);
    }

    return result;
  }

  if (not std::holds_alternative<
          ShapeMapOptions<IncludeTransitionEndsAtCube, Object>>(
          shape_map_options)) {
    ERROR("Unknown ShapeMap.");
  }

  const auto& hard_coded_options =
      std::get<ShapeMapOptions<IncludeTransitionEndsAtCube, Object>>(
          shape_map_options);

  std::array<DataVector, 3> shape_funcs =
      make_array<3, DataVector>(shape_zeros);
  std::array<DataVector, 4> size_funcs =
      make_array<4, DataVector>(DataVector{1, 0.0});

  if (hard_coded_options.initial_values.has_value()) {
    if (std::holds_alternative<KerrSchildFromBoyerLindquist>(
            hard_coded_options.initial_values.value())) {
      const ylm::Spherepack ylm{hard_coded_options.l_max,
                                hard_coded_options.l_max};
      const auto& mass_and_spin = std::get<KerrSchildFromBoyerLindquist>(
          hard_coded_options.initial_values.value());
      const DataVector radial_distortion =
          deformed_radius -
          get(gr::Solutions::kerr_schild_radius_from_boyer_lindquist(
              deformed_radius, ylm.theta_phi_points(), mass_and_spin.mass,
              mass_and_spin.spin));
      shape_funcs[0] = ylm.phys_to_spec(radial_distortion);
      // Transform from SPHEREPACK to actual Ylm for size func
      size_funcs[0][0] = shape_funcs[0][0] * sqrt(0.5 * M_PI);
      // Set l=0 for shape map to 0 because size control will adjust l=0
      shape_funcs[0][0] = 0.0;
    } else if (std::holds_alternative<YlmsFromFile>(
                   hard_coded_options.initial_values.value())) {
      const auto& files =
          std::get<YlmsFromFile>(hard_coded_options.initial_values.value());
      const std::string& h5_filename = files.h5_filename;
      const std::vector<std::string>& subfile_names = files.subfile_names;
      const double match_time = files.match_time;
      const double match_time_epsilon =
          files.match_time_epsilon.value_or(1e-12);
      const bool set_l1_coefs_to_zero = files.set_l1_coefs_to_zero;
      ylm::SpherepackIterator iter{l_max, l_max};

      for (size_t i = 0; i < subfile_names.size(); i++) {
        // Frame doesn't matter here
        const ylm::Strahlkorper<Frame::Distorted> file_strahlkorper =
            ylm::read_surface_ylm_single_time<Frame::Distorted>(
                h5_filename, gsl::at(subfile_names, i), match_time,
                match_time_epsilon, files.check_frame);
        const ylm::Strahlkorper<Frame::Distorted> this_strahlkorper{
            hard_coded_options.l_max, 1.0, std::array{0.0, 0.0, 0.0}};

        // The coefficients in the shape map are stored as the negative
        // coefficients of the strahlkorper, so we need to multiply by -1 here.
        gsl::at(shape_funcs, i) =
            -1.0 * file_strahlkorper.ylm_spherepack().prolong_or_restrict(
                       file_strahlkorper.coefficients(),
                       this_strahlkorper.ylm_spherepack());
        // Transform from SPHEREPACK to actual Ylm for size func
        gsl::at(size_funcs, i)[0] =
            gsl::at(shape_funcs, i)[0] * sqrt(0.5 * M_PI) +
            // Account for the size of the original sphere, since the shape/size
            // coefficients are deformations from the original sphere.
            // The factor 2 sqrt(pi) is 1/Y_00.
            deformed_radius * 2.0 * sqrt(M_PI);
        // Set l=0 for shape map to 0 because size control will adjust l=0
        gsl::at(shape_funcs, i)[0] = 0.0;
        if (set_l1_coefs_to_zero) {
          for (int m = -1; m <= 1; m++) {
            gsl::at(shape_funcs, i)[iter.set(1_st, m)()] = 0.0;
          }
        }
      }
    } else if (std::holds_alternative<YlmsFromSpEC>(
                   hard_coded_options.initial_values.value())) {
      const auto& spec_option =
          std::get<YlmsFromSpEC>(hard_coded_options.initial_values.value());
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
      std::optional<size_t> file_l_max{};
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

        if (file_l_max.has_value()) {
          ERROR("Found more than one time in the SpEC dat file "
                << dat_filename << " that is within a relative epsilon of "
                << match_time_epsilon << " of the time requested " << time);
        }

        // Casting to an integer floors a double, so we add 0.5 before we take
        // the sqrt to avoid any rounding issues
        const auto file_l_max_plus_one =
            static_cast<size_t>(sqrt(static_cast<double>(total_col) + 0.5));
        if (file_l_max_plus_one == 0) {
          ERROR(
              "Invalid l_max from SpEC dat file. l_max + 1 was computed to be "
              "0");
        }
        file_l_max = file_l_max_plus_one - 1;

        ss >> center[0];
        ss >> center[1];
        ss >> center[2];

        coefficients.destructive_resize(ylm::Spherepack::spectral_size(
            file_l_max.value(), file_l_max.value()));

        file_iter =
            ylm::SpherepackIterator{file_l_max.value(), file_l_max.value()};

        for (int l = 0; l <= static_cast<int>(file_l_max.value()); l++) {
          for (int m = -l; m <= l; m++) {
            ss >> coefficients[file_iter.set(static_cast<size_t>(l), m)()];
          }
        }
      }

      if (not file_l_max.has_value()) {
        ERROR_NO_TRACE("Unable to find requested time "
                       << time << " within an epsilon of " << match_time_epsilon
                       << " in SpEC dat file " << dat_filename);
      }

      const ylm::Strahlkorper<Frame::Inertial> file_strahlkorper{
          file_l_max.value(), file_l_max.value(), coefficients, center};
      const ylm::Strahlkorper<Frame::Inertial> this_strahlkorper{
          l_max, 1.0, std::array{0.0, 0.0, 0.0}};
      ylm::SpherepackIterator iter{l_max, l_max};

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
    }
  }

  // If any size options were specified, those override the values from the
  // shape coefs
  if (hard_coded_options.initial_size_values.has_value()) {
    for (size_t i = 0; i < 3; i++) {
      gsl::at(size_funcs, i)[0] =
          gsl::at(hard_coded_options.initial_size_values.value(), i);
    }
  }

  result[shape_name] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, std::move(shape_funcs), shape_expiration_time);
  result[size_name] = std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
      initial_time, std::move(size_funcs), size_expiration_time);

  return result;
}

#define OBJECT(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INCLUDETRANSITION(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                          \
  template size_t l_max_from_shape_options(                           \
      const std::variant<                                             \
          ShapeMapOptions<INCLUDETRANSITION(data), OBJECT(data)>,     \
          FromVolumeFileShapeSize<OBJECT(data)>>& shape_map_options); \
  template bool transition_ends_at_cube_from_shape_options(           \
      const std::variant<                                             \
          ShapeMapOptions<INCLUDETRANSITION(data), OBJECT(data)>,     \
          FromVolumeFileShapeSize<OBJECT(data)>>& shape_map_options); \
  template FunctionsOfTimeMap get_shape_and_size(                     \
      const std::variant<                                             \
          ShapeMapOptions<INCLUDETRANSITION(data), OBJECT(data)>,     \
          FromVolumeFileShapeSize<OBJECT(data)>>& shape_map_options,  \
      double initial_time, double shape_expiration_time,              \
      double size_expiration_time, double deformed_radius);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (domain::ObjectLabel::A, domain::ObjectLabel::B,
                         domain::ObjectLabel::None),
                        (true, false))

#undef INCLUDETRANSITION
#undef INSTANTIATE

#define INSTANTIATE(_, data) \
  template class FromVolumeFileShapeSize<OBJECT(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (domain::ObjectLabel::A, domain::ObjectLabel::B,
                         domain::ObjectLabel::None))

#undef OBJECT
#undef INSTANTIATE
}  // namespace domain::creators::time_dependent_options
