// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeModeRecorder.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/Actions/GetLockPointer.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
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

namespace intrp::callbacks {
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
 * - `gh::Tags::Pi`
 * - `gh::Tags::Phi`
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

  using gh_source_vars_for_cce =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>;

  using gh_source_vars_from_interpolation =
      typename InterpolationTargetTag::vars_to_interpolate_to_target;

  static_assert(
      std::is_same_v<tmpl::list_difference<
                         Cce::Tags::worldtube_boundary_tags_for_writing<>,
                         cce_boundary_tags>,
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
    const double time =
        intrp::InterpolationTarget_detail::get_temporal_id_value(temporal_id);
    const auto& sphere =
        Parallel::get<Tags::Sphere<InterpolationTargetTag>>(cache);
    const auto& filename_prefix = Parallel::get<Cce::Tags::FilePrefix>(cache);

    if (sphere.angular_ordering != ylm::AngularOrdering::Cce) {
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
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(all_gh_vars);
    const auto& all_pi = get<gh::Tags::Pi<DataVector, 3>>(all_gh_vars);
    const auto& all_phi = get<gh::Tags::Phi<DataVector, 3>>(all_gh_vars);

    const tnsr::aa<DataVector, 3, ::Frame::Inertial> spacetime_metric;
    const tnsr::aa<DataVector, 3, ::Frame::Inertial> pi;
    const tnsr::iaa<DataVector, 3, ::Frame::Inertial> phi;

    // Bondi data
    Variables<cce_boundary_tags> bondi_boundary_data{num_points_single_sphere};

    // Even though no other cores should be writing to this file, we
    // still need to get the h5 file lock so the system hdf5 doesn't get
    // upset
    auto* hdf5_lock = Parallel::local_branch(
                          Parallel::get_parallel_component<
                              observers::ObserverWriter<Metavariables>>(cache))
                          ->template local_synchronous_action<
                              observers::Actions::GetLockPointer<
                                  observers::Tags::H5FileLock>>();

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
                       << std::setw(4) << std::lround(radius) << ".h5";

      // Lock now and it'll be unlocked for this radius after we finish writing
      // the data to disk
      const std::lock_guard lock(*hdf5_lock);
      Cce::WorldtubeModeRecorder recorder{l_max, filename};

      tmpl::for_each<Cce::Tags::worldtube_boundary_tags_for_writing<>>(
          [&](auto tag_v) {
            using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
            constexpr int spin = tag::tag::type::type::spin;

            const ComplexDataVector& bondi_nodal_data =
                get(get<tag>(bondi_boundary_data)).data();

            recorder.append_modal_data<spin>(
                Cce::dataset_label_for_tag<typename tag::tag>(), time,
                bondi_nodal_data);
          });
    }
  }
};
}  // namespace intrp::callbacks
