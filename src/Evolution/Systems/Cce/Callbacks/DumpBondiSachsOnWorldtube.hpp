// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/AngularOrdering.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
namespace callbacks {
/*!
 * \brief Post interpolation callback that dumps metric data in Bondi-Sachs form
 * on a number of extraction radii given by the `intrp::TargetPoints::Sphere`
 * target.
 *
 * To use this callback, the target must be the `intrp::TargetPoints::Sphere`
 * target in the inertial frame. This callback also expects that the GH source
 * vars on each of the target spheres are:
 *
 * - `gr::Tags::SpacetimeMetric`
 * - `GeneralizedHarmonic::Tags::Pi`
 * - `GeneralizedHarmonic::Tags::Phi`
 *
 * This callback will write a new `H5` file for each extraction radius in the
 * Sphere target. The name of this file will be a file prefix specified by the
 * Cce::Tags::FilePrefix prepended onto `CceRXXXX.h5` where the XXXX is the
 * zero-padded extraction radius rounded to the nearest integer. The quantities
 * that will be written are
 *
 * - `Cce::Tags::BondiBeta`
 * - `Cce::Tags::Dr<Cce::Tags::BondiJ>`
 * - `Cce::Tags::Du<Cce::Tags::BondiR>`
 * - `Cce::Tags::BondiH`
 * - `Cce::Tags::BondiJ`
 * - `Cce::Tags::BondiQ`
 * - `Cce::Tags::BondiR`
 * - `Cce::Tags::BondiU`
 * - `Cce::Tags::BondiW`
 *
 * \note For all real quantities (Beta, DuR, R, W) we omit writing the
 * negative m modes, and the imaginary part of the m = 0 mode.
 */
template <typename InterpolationTargetTag>
struct DumpBondiSachsOnWorldtube
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  static constexpr double fill_invalid_points_with =
      std::numeric_limits<double>::quiet_NaN();

  using const_global_cache_tags = tmpl::list<Cce::Tags::FilePrefix>;

  using cce_boundary_tags = Cce::Tags::characteristic_worldtube_boundary_tags<
      Cce::Tags::BoundaryValue>;

  using cce_tags_to_dump = db::wrap_tags_in<
      Cce::Tags::BoundaryValue,
      tmpl::list<Cce::Tags::BondiBeta, Cce::Tags::Dr<Cce::Tags::BondiJ>,
                 Cce::Tags::Du<Cce::Tags::BondiR>, Cce::Tags::BondiH,
                 Cce::Tags::BondiJ, Cce::Tags::BondiQ, Cce::Tags::BondiR,
                 Cce::Tags::BondiU, Cce::Tags::BondiW>>;

  using gh_source_vars_for_cce =
      tmpl::list<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>;

  using gh_source_vars_from_interpolation =
      typename InterpolationTargetTag::vars_to_interpolate_to_target;

  static_assert(
      std::is_same_v<tmpl::list_difference<cce_tags_to_dump, cce_boundary_tags>,
                     tmpl::list<>>,
      "Cce tags to dump are not in the boundary tags.");

  static_assert(
      tmpl::and_<
          std::is_same<tmpl::list_difference<gh_source_vars_from_interpolation,
                                             gh_source_vars_for_cce>,
                       tmpl::list<>>,
          std::is_same<tmpl::list_difference<gh_source_vars_for_cce,
                                             gh_source_vars_from_interpolation>,
                       tmpl::list<>>>::type::value,
      "To use DumpBondiSachsOnWorldtube, the GH source variables must be the "
      "spacetime metric, pi, and phi.");

  static_assert(
      std::is_same_v<typename InterpolationTargetTag::compute_target_points,
                     intrp::TargetPoints::Sphere<InterpolationTargetTag,
                                                 ::Frame::Inertial>>,
      "To use the DumpBondiSachsOnWorltube post interpolation callback, you "
      "must use the intrp::TargetPoints::Sphere target in the inertial "
      "frame");

  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    const auto& sphere =
        Parallel::get<Tags::Sphere<InterpolationTargetTag>>(cache);
    const auto& filename_prefix = Parallel::get<Cce::Tags::FilePrefix>(cache);

    if (sphere.angular_ordering != intrp::AngularOrdering::Cce) {
      ERROR(
          "To use the DumpBondiSachsOnWorldtube post interpolation callback, "
          "the angular ordering of the Spheres must be Cce, not "
          << sphere.angular_ordering);
    }

    const auto& radii = sphere.radii;
    const size_t l_max = sphere.l_max;
    const size_t num_points_single_sphere =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    const auto& all_gh_vars =
        db::get<::Tags::Variables<gh_source_vars_from_interpolation>>(box);

    const auto& all_spacetime_metric =
        get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial>>(all_gh_vars);
    const auto& all_pi =
        get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(all_gh_vars);
    const auto& all_phi =
        get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(all_gh_vars);

    const tnsr::aa<DataVector, 3, ::Frame::Inertial> spacetime_metric;
    const tnsr::aa<DataVector, 3, ::Frame::Inertial> pi;
    const tnsr::iaa<DataVector, 3, ::Frame::Inertial> phi;

    // Bondi data
    Variables<cce_boundary_tags> bondi_boundary_data{num_points_single_sphere};
    ComplexModalVector goldberg_mode_buffer{square(l_max + 1)};
    const std::vector<std::string> all_legend = build_legend(l_max, false);
    const std::vector<std::string> real_legend = build_legend(l_max, true);

    size_t offset = 0;
    for (const auto& radius : radii) {
      // Set data references so we don't copy data unnecessarily
      for (size_t a = 0; a < 4; a++) {
        for (size_t b = 0; b < 4; b++) {
          make_const_view(make_not_null(&spacetime_metric.get(a, b)),
                          all_spacetime_metric.get(a, b), offset,
                          num_points_single_sphere);
          make_const_view(make_not_null(&pi.get(a, b)), all_pi.get(a, b),
                          offset, num_points_single_sphere);
          for (size_t i = 0; i < 3; i++) {
            make_const_view(make_not_null(&phi.get(i, a, b)),
                            all_phi.get(i, a, b), offset,
                            num_points_single_sphere);
          }
        }
      }

      offset += num_points_single_sphere;

      Cce::create_bondi_boundary_data(make_not_null(&bondi_boundary_data), phi,
                                      pi, spacetime_metric, radius, l_max);

      const std::string filename =
          MakeString{} << filename_prefix << "CceR" << std::setfill('0')
                       << std::setw(4) << std::lround(radius);

      tmpl::for_each<cce_tags_to_dump>(
          [&temporal_id, &l_max, &all_legend, &real_legend, &filename,
           &bondi_boundary_data, &goldberg_mode_buffer, &cache](auto tag_v) {
            using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
            constexpr int spin = tag::tag::type::type::spin;
            // Spin = 0 does not imply a quantity is real. However, all the tags
            // we want to print out that have spin = 0 happen to be real, so we
            // use this as an indicator.
            constexpr bool is_real = spin == 0;

            const auto& legend = is_real ? real_legend : all_legend;

            // `tag` is a BoundaryValue. We want the actual tag name
            const std::string subfile_name{
                "/" + replace_name(db::tag_name<typename tag::tag>())};

            const auto& bondi_data = get(get<tag>(bondi_boundary_data));

            // Convert our modal data to goldberg modes
            SpinWeighted<ComplexModalVector, spin> goldberg_modes;
            goldberg_modes.set_data_ref(make_not_null(&goldberg_mode_buffer));
            Spectral::Swsh::libsharp_to_goldberg_modes(
                make_not_null(&goldberg_modes),
                Spectral::Swsh::swsh_transform(l_max, 1, bondi_data), l_max);

            std::vector<double> data_to_write_buffer;
            data_to_write_buffer.reserve(number_of_components(l_max, is_real));
            data_to_write_buffer.emplace_back(
                intrp::InterpolationTarget_detail::get_temporal_id_value(
                    temporal_id));

            // We loop over ell and m rather than just the total number of modes
            // because we don't print negative m or the imaginary part of m=0
            // for real quantities.
            for (size_t ell = 0; ell <= l_max; ell++) {
              for (int m = is_real ? 0 : -static_cast<int>(ell);
                   m <= static_cast<int>(ell); m++) {
                const size_t goldberg_index =
                    Spectral::Swsh::goldberg_mode_index(l_max, ell, m);
                data_to_write_buffer.push_back(
                    real(goldberg_modes.data()[goldberg_index]));
                if (not is_real or m != 0) {
                  data_to_write_buffer.push_back(
                      imag(goldberg_modes.data()[goldberg_index]));
                }
              }
            }

            ASSERT(legend.size() == data_to_write_buffer.size(),
                   "Legend (" << legend.size()
                              << ") does not have the same number of "
                                 "components as data to write ("
                              << data_to_write_buffer.size() << ") for tag "
                              << db::tag_name<typename tag::tag>());

            observers::ThreadedActions::ReductionActions_detail::write_data(
                subfile_name, observers::input_source_from_cache(cache), legend,
                std::make_tuple(data_to_write_buffer), filename,
                std::index_sequence<0>{});
          });
    }
  }

 private:
  // There are exactly half the number of modes for spin = 0 quantities as their
  // are for spin != 0
  static size_t number_of_components(const size_t l_max, const bool is_real) {
    return 1 + square(l_max + 1) * (is_real ? 1 : 2);
  }

  static std::vector<std::string> build_legend(const size_t l_max,
                                               const bool is_real) {
    std::vector<std::string> legend;
    legend.reserve(number_of_components(l_max, is_real));
    legend.emplace_back("Time");
    for (int ell = 0; ell <= static_cast<int>(l_max); ++ell) {
      for (int m = is_real ? 0 : -ell; m <= ell; ++m) {
        legend.push_back(MakeString{} << "Re(" << ell << "," << m << ")");
        // For real quantities, don't include the imaginary m=0
        if (not is_real or m != 0) {
          legend.push_back(MakeString{} << "Im(" << ell << "," << m << ")");
        }
      }
    }
    return legend;
  }

  // These match names that CCE executable expects
  static std::string replace_name(const std::string& db_tag_name) {
    if (db_tag_name == "BondiBeta") {
      return "Beta";
    } else if (db_tag_name == "Dr(J)") {
      return "DrJ";
    } else if (db_tag_name == "Du(R)") {
      return "DuR";
    } else {
      return db_tag_name;
    }
  }
};
}  // namespace callbacks
}  // namespace intrp
