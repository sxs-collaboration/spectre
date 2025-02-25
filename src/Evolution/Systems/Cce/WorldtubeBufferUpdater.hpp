// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Version.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace Tags::detail {
// tags for use in the buffers for the input worldtube data management classes
template <typename T>
using SpatialMetric = gr::Tags::SpatialMetric<T, 3>;
template <typename T>
using Shift = gr::Tags::Shift<T, 3>;
template <typename T>
using Lapse = gr::Tags::Lapse<T>;
// This is B^i in the 3+1 equations
template <typename T>
struct AuxiliaryShift : db::SimpleTag {
  using type = tnsr::I<T, 3, ::Frame::Inertial>;
};
// This is \Gamma^i in the Z4c equations
template <typename T>
struct ConformalChristoffel : db::SimpleTag {
  using type = tnsr::I<T, 3, ::Frame::Inertial>;
};
template <typename T>
using ExtrinsicCurvature = gr::Tags::ExtrinsicCurvature<T, 3>;

// The three metric quantities we read in from disk (no derivatives)
template <typename T>
using metric_tags = tmpl::list<SpatialMetric<T>, Shift<T>, Lapse<T>>;

// radial derivative prefix tag to be used with the input worldtube data
template <typename Tag>
struct Dr : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};
// For cartesian derivs. Each dataset is stored separately in the file, so it
// doesn't make sense to use ::Tags::deriv here
template <typename Tag>
struct Dx : db::SimpleTag, db::PrefixTag {
  static constexpr size_t index = 0;
  using type = typename Tag::type;
};
template <typename Tag>
struct Dy : db::SimpleTag, db::PrefixTag {
  static constexpr size_t index = 1;
  using type = typename Tag::type;
};
template <typename Tag>
struct Dz : db::SimpleTag, db::PrefixTag {
  static constexpr size_t index = 2;
  using type = typename Tag::type;
};
template <typename Tag>
using cartesian_derivs_t = tmpl::list<Dx<Tag>, Dy<Tag>, Dz<Tag>>;

// tag for the string for accessing the quantity associated with `Tag` in
// worldtube h5 file
template <typename Tag>
struct InputDataSet : db::SimpleTag, db::PrefixTag {
  using type = std::string;
  using tag = Tag;
};

// Puts `Tag`, `::Tags::dt<Tag>`, and `Cce::Tags::Dr<Tag>` into a `tmpl::list`
template <typename Tag>
struct apply_derivs {
  using type = tmpl::list<Tag, ::Tags::dt<Tag>, Dr<Tag>>;
};
template <typename Tag>
using apply_derivs_t = typename apply_derivs<Tag>::type;

// Puts `Tag`, `::Tags::dt<Tag>`, `Dx<Tag>`, `Dy<Tag>`, `Dz<Tag>` into a
// `tmpl::list`
template <typename Tag>
struct apply_derivs_adm {
  using type =
      tmpl::flatten<tmpl::list<Tag, ::Tags::dt<Tag>, cartesian_derivs_t<Tag>>>;
};
template <typename Tag>
using apply_derivs_adm_t = typename apply_derivs_adm<Tag>::type;
}  // namespace Tags::detail

namespace detail {
// generates the component dataset name in the worldtube file based on the
// tensor indices requested. For instance, if called with arguments ("/g", 0,1),
// it returns the dataset name "/gxy".
template <typename... T>
std::string dataset_name_for_component(std::string base_name,
                                       const T... indices) {  // NOLINT
  const auto add_index = [&base_name](size_t index) {
    ASSERT(index < 3, "The character-arithmetic index must be less than 3.");
    base_name += static_cast<char>('x' + index);
  };
  EXPAND_PACK_LEFT_TO_RIGHT(add_index(indices));
  // void cast so that compilers can tell it's used.
  (void)add_index;
  return base_name;
}

// creates a pair of indices such that the difference is `2 *
// interpolator_length + pad`, centered around `time`, and bounded by
// `lower_bound` and `upper_bound`. If it cannot be centered, it gives a span
// that is appropriately sized and bounded by the supplied bounds. If the bounds
// are too constraining for the necessary size, it gives a span that is the
// correct size starting at `lower bound`, but not constrained by `upper_bound`
std::pair<size_t, size_t> create_span_for_time_value(
    double time, size_t pad, size_t interpolator_length, size_t lower_bound,
    size_t upper_bound, const DataVector& time_buffer);

// retrieves time stamps and lmax the from the specified file. the bools
// `is_real`, `is_modal_data`, and `is_complex` are used to predict the number
// of columns in the dat file and possibly error if that number is incorrect.
// Note that if `is_modal_data` is true, then we use `is_real` but `is_complex`
// is unneeded. If `is_modal_data` if false, then `is_complex` is used, but
// `is_real` is unneeded.
void set_time_buffer_and_lmax(gsl::not_null<DataVector*> time_buffer,
                              size_t& l_max, const h5::Dat& data, bool is_real,
                              bool is_modal_data, bool is_complex);

// updates `time_span_start` and `time_span_end` based on the provided `time`,
// and inserts the cooresponding modal data (for `InputTags`) from worldtube H5
// file into `buffers`. The function is used by Bondi and Klein-Gordon systems.
template <typename InputTags>
double update_buffers_for_time(
    gsl::not_null<Variables<InputTags>*> buffers,
    gsl::not_null<size_t*> time_span_start,
    gsl::not_null<size_t*> time_span_end, double time, size_t computation_l_max,
    size_t l_max, size_t interpolator_length, size_t buffer_depth,
    const DataVector& time_buffer,
    const tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<Tags::detail::InputDataSet, InputTags>>& dataset_names,
    const h5::H5File<h5::AccessType::ReadOnly>& cce_data_file);
}  // namespace detail

/// the full set of metric tensors to be extracted from the worldtube h5 file
template <typename T>
using cce_metric_input_tags =
    tmpl::flatten<tmpl::transform<Tags::detail::metric_tags<T>,
                                  Tags::detail::apply_derivs<tmpl::_1>>>;

template <typename T>
using cce_metric_adm_input_tags = tmpl::flatten<tmpl::push_back<
    tmpl::transform<Tags::detail::metric_tags<T>,
                    Tags::detail::apply_derivs_adm<tmpl::_1>>,
    Tags::detail::ExtrinsicCurvature<T>, Tags::detail::AuxiliaryShift<T>,
    Tags::detail::ConformalChristoffel<T>>>;

using klein_gordon_input_tags =
    tmpl::list<Spectral::Swsh::Tags::SwshTransform<Tags::KleinGordonPsi>,
               Spectral::Swsh::Tags::SwshTransform<Tags::KleinGordonPi>>;

/// \cond
template <typename T>
class MetricWorldtubeH5BufferUpdater;
template <typename T>
class BondiWorldtubeH5BufferUpdater;
class KleinGordonWorldtubeH5BufferUpdater;
/// \endcond

/*!
 *  \brief Abstract base class for utilities that are able to perform the buffer
 *  updating procedure needed by the `WorldtubeDataManager` or by the
 *  `PreprocessCceWorldtube` executable.
 *
 *  \details The methods that are required to be overridden in the derived
 * classes are:
 *  - `WorldtubeBufferUpdater::update_buffers_for_time()`:
 *  updates the buffers passed by pointer and the `time_span_start` and
 *  `time_span_end` to be appropriate for the requested `time`,
 *  `interpolator_length`, and `buffer_depth`.
 *  - `WorldtubeBufferUpdater::get_clone()`
 *  clone function to obtain a `std::unique_ptr` of the base
 *  `WorldtubeBufferUpdater`, needed to pass around the factory-created
 *  object.
 *  - `WorldtubeBufferUpdater::time_is_outside_range()`
 *  the override should return `true` if the `time` could be used in a
 *  `update_buffers_for_time` call given the data available to the derived
 *  class, and `false` otherwise
 *  - `WorldtubeBufferUpdater::get_l_max()`
 *  The override should return the `l_max` it uses in the
 *  Goldberg modal data placed in the buffers.
 *  - `WorldtubeBufferUpdater::get_extraction_radius()`
 *  The override should return the coordinate radius associated with the modal
 *  worldtube data that it supplies in the buffer update function. This is
 *  currently assumed to be a single double, but may be generalized in future
 *  to be time-dependent.
 *  - `WorldtubeBufferUpdater::get_time_buffer`
 *  The override should return the vector of times that it can produce
 *  data at. For instance, if associated with a file input, this will be the
 *  times at each of the rows of the time-series data.
 */
template <typename BufferTags>
class WorldtubeBufferUpdater : public PUP::able {
 public:
  using creatable_classes =
      tmpl::list<MetricWorldtubeH5BufferUpdater<ComplexModalVector>,
                 MetricWorldtubeH5BufferUpdater<DataVector>,
                 BondiWorldtubeH5BufferUpdater<ComplexModalVector>,
                 BondiWorldtubeH5BufferUpdater<ComplexDataVector>,
                 KleinGordonWorldtubeH5BufferUpdater>;

  WRAPPED_PUPable_abstract(WorldtubeBufferUpdater);  // NOLINT

  virtual double update_buffers_for_time(
      gsl::not_null<Variables<BufferTags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length, size_t buffer_depth,
      bool time_varies_fastest = true) const = 0;

  virtual std::unique_ptr<WorldtubeBufferUpdater> get_clone() const = 0;

  virtual bool time_is_outside_range(double time) const = 0;

  virtual size_t get_l_max() const = 0;

  virtual double get_extraction_radius() const = 0;

  virtual bool has_version_history() const = 0;

  virtual DataVector& get_time_buffer() = 0;
};

/*!
 * \brief A `WorldtubeBufferUpdater` specialized to CCE input worldtube H5 files
 * that have cartesian metric components stored in either modal or nodal form.
 *
 * \details To read in modal data, template this class as
 * `MetricWorldtubeH5BufferUpdater<ComplexModalVector>`. To read in nodal data,
 * template the class as `MetricWorldtubeH5BufferUpdater<DataVector>`. This
 * class also has the ability to read in data specifically written by SpEC.
 */
template <typename T>
class MetricWorldtubeH5BufferUpdater
    : public WorldtubeBufferUpdater<cce_metric_input_tags<T>> {
  static_assert(std::is_same_v<T, ComplexModalVector> or
                    std::is_same_v<T, DataVector>,
                "Can only use ComplexModalVector or DataVector in a "
                "MetricWorldtubeH5BufferUpdater.");

 public:
  static constexpr bool is_modal = std::is_same_v<T, ComplexModalVector>;

  /*!
   * \brief Options needed when reading in the extrinsic curvature and auxiliary
   * shift (BSSN) or the trace of the conformal christoffel (Z4c) from a non-GH
   * evolution code.
   *
   * \details Can also be used as an option tag
   */
  struct AdmOptions {
    static std::string name() { return "AdmMetricNodal"; }

    struct Advective {
      using type = bool;
      static constexpr Options::String help =
          "Add advective term to time derivative.";
    };

    struct Lapse {
      using type = Lapse;
      Lapse() = default;
      using options = tmpl::list<Advective>;
      static constexpr Options::String help =
          "Options for 1+log slicing of the lapse.";
      explicit Lapse(const bool advective) : is_advective(advective) {}

      void pup(PUP::er& p) { p | is_advective; }

      bool is_advective;
    };

    struct Shift {
      using type = Shift;
      struct SecondOrderDriverEta {
        using type = double;
        static constexpr Options::String help =
            "Factor 'eta' in front of shift vector in Eq. 12 of Hilditch 2013 "
            "(typically 2/M). To use this, the trace conformal christoffel "
            "must be dumped. Here mu_S is assumed to be 1/lapse^2.";
      };
      struct FirstOrderDriverFactor {
        using type = double;
        static constexpr Options::String help =
            "Factor in front of auxiliary shift vector B^i in left Eq. 4.89 of "
            "B&S (typically 0.75). To use this, the auxiliary shift vector B^i "
            "must be dumped. Here mu_S is assumed to be 1/lapse^2.";
      };
      Shift() = default;
      using options =
          tmpl::list<Advective,
                     Options::Alternatives<tmpl::list<SecondOrderDriverEta>,
                                           tmpl::list<FirstOrderDriverFactor>>>;

      static constexpr Options::String help =
          "Options for Gamma driver shift condition.";
      Shift(tmpl::list<Advective, SecondOrderDriverEta> /*meta*/,
            const bool advective, const double factor)
          : is_advective(advective),
            is_first_order(false),
            extra_factor(factor) {}
      Shift(tmpl::list<Advective, FirstOrderDriverFactor> /*meta*/,
            const bool advective, const double factor)
          : is_advective(advective),
            is_first_order(true),
            extra_factor(factor) {}

      void pup(PUP::er& p) {
        p | is_advective;
        p | is_first_order;
        p | extra_factor;
      }

      bool is_advective;
      bool is_first_order;
      double extra_factor;
    };

    using options = tmpl::list<Lapse, Shift>;
    static constexpr Options::String help =
        "Gauge options when reading in extrinsic curvature additional variable "
        "from non-GH evolutions.";

    AdmOptions() = default;
    AdmOptions(const Lapse& lapse_in, const Shift& shift_in)
        : lapse(lapse_in), shift(shift_in) {}

    void pup(PUP::er& p) {
      p | lapse;
      p | shift;
    }

    Lapse lapse;
    Shift shift;
  };

  // charm needs the empty constructor
  MetricWorldtubeH5BufferUpdater() = default;

  /// The constructor takes the filename of the H5 file that will be used
  /// for boundary data. The extraction radius can either be passed in directly,
  /// or if it takes the value `std::nullopt`, then the extraction radius is
  /// retrieved as an integer in the filename. Also the user can specify if the
  /// H5 file was written by SpEC or not, because SpEC has some different
  /// conventions than we use here.
  explicit MetricWorldtubeH5BufferUpdater(
      const std::string& cce_data_filename,
      std::optional<double> extraction_radius = std::nullopt,
      bool descending_m = true,
      const std::optional<AdmOptions>& adm_options = std::nullopt);

  // NOLINTNEXTLINE
  WRAPPED_PUPable_decl_base_template(
      WorldtubeBufferUpdater<cce_metric_input_tags<T>>,
      MetricWorldtubeH5BufferUpdater);

  explicit MetricWorldtubeH5BufferUpdater(CkMigrateMessage* /*unused*/) {}

  /// \brief Update the `buffers`, `time_span_start`, and `time_span_end` with
  /// data (either Goldberg modal data or just nodal data depending on the
  /// template parameter to this class) and the start and end index in the
  /// member `time_buffer_` covered by the newly updated `buffers`.
  ///
  /// The function returns the next time at which a full update will occur. If
  /// called again at times earlier than the next full update time, it will
  /// leave the `buffers` unchanged and again return the next needed time.
  double update_buffers_for_time(
      gsl::not_null<Variables<cce_metric_input_tags<T>>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length, size_t buffer_depth,
      bool time_varies_fastest = true) const override;

  std::unique_ptr<WorldtubeBufferUpdater<cce_metric_input_tags<T>>> get_clone()
      const override;

  /// The time can only be supported in the buffer update if it is between the
  /// first and last time of the input file.
  bool time_is_outside_range(double time) const override;

  /// retrieves the l_max of the input file
  size_t get_l_max() const override { return l_max_; }

  /// retrieves the extraction radius
  double get_extraction_radius() const override { return extraction_radius_; }

  /// The time buffer is supplied by non-const reference to allow views to
  /// easily point into the buffer.
  ///
  /// \warning Altering this buffer outside of the constructor of this class
  /// results in undefined behavior! This should be supplied by const reference
  /// once there is a convenient method of producing a const view of a vector
  /// type.
  DataVector& get_time_buffer() override { return time_buffer_; }

  bool has_version_history() const override { return has_version_history_; }

  /// Serialization for Charm++.
  void pup(PUP::er& p) override;

 private:
  void update_radial_formulation(
      gsl::not_null<Variables<cce_metric_input_tags<T>>*> buffers,
      size_t computation_l_max, size_t time_span_start, size_t time_span_end,
      bool time_varies_fastest) const;
  template <typename U = T>
  typename std::enable_if_t<std::is_same_v<U, DataVector>>
  update_adm_formulation(
      gsl::not_null<Variables<cce_metric_input_tags<DataVector>>*> buffers,
      size_t computation_l_max, size_t time_span_start, size_t time_span_end,
      bool time_varies_fastest) const;
  void update_buffer(gsl::not_null<T*> buffer_to_update,
                     const h5::Dat& read_data, size_t computation_l_max,
                     size_t time_span_start, size_t time_span_end,
                     bool time_varies_fastest) const;

  bool has_version_history_ = true;
  double extraction_radius_ = std::numeric_limits<double>::signaling_NaN();
  size_t l_max_ = 0;

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file_;
  std::string filename_;
  bool descending_m_ = true;
  std::optional<AdmOptions> adm_options_;

  tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
      Tags::detail::InputDataSet,
      tmpl::remove_duplicates<tmpl::append<cce_metric_input_tags<T>,
                                           cce_metric_adm_input_tags<T>>>>>
      dataset_names_;

  // stores all the times in the input file
  DataVector time_buffer_;
};

/*!
 * \brief A `WorldtubeBufferUpdater` specialized to CCE input worldtube H5 files
 * that have metric data stored in Bondi-Sachs format in either either modal or
 * nodal form.
 *
 * \details To read in modal data, template this class as
 * `MetricWorldtubeH5BufferUpdater<ComplexModalVector>`. To read in nodal data,
 * template the class as `MetricWorldtubeH5BufferUpdater<ComplexDataVector>`.
 */
template <typename T>
class BondiWorldtubeH5BufferUpdater
    : public WorldtubeBufferUpdater<tmpl::conditional_t<
          std::is_same_v<T, ComplexModalVector>,
          Tags::worldtube_boundary_tags_for_writing<
              Spectral::Swsh::Tags::SwshTransform>,
          Tags::worldtube_boundary_tags_for_writing<Tags::BoundaryValue>>> {
  static_assert(std::is_same_v<T, ComplexModalVector> or
                    std::is_same_v<T, ComplexDataVector>,
                "Can only use ComplexModalVector or ComplexDataVector in a "
                "BondiWorldtubeH5BufferUpdater.");

 public:
  using tags_for_writing = tmpl::conditional_t<
      std::is_same_v<T, ComplexModalVector>,
      Tags::worldtube_boundary_tags_for_writing<
          Spectral::Swsh::Tags::SwshTransform>,
      Tags::worldtube_boundary_tags_for_writing<Tags::BoundaryValue>>;

  static constexpr bool is_modal = std::is_same_v<T, ComplexModalVector>;

  // charm needs the empty constructor
  BondiWorldtubeH5BufferUpdater() = default;

  /// The constructor takes the filename of the H5 file that will be used
  /// for boundary data. The extraction radius can either be passed in directly,
  /// or if it takes the value `std::nullopt`, then the extraction radius is
  /// retrieved as an integer in the filename.
  explicit BondiWorldtubeH5BufferUpdater(
      const std::string& cce_data_filename,
      std::optional<double> extraction_radius = std::nullopt);

  // NOLINTNEXTLINE
  WRAPPED_PUPable_decl_base_template(
      SINGLE_ARG(WorldtubeBufferUpdater<tags_for_writing>),
      BondiWorldtubeH5BufferUpdater);

  explicit BondiWorldtubeH5BufferUpdater(CkMigrateMessage* /*unused*/) {}

  /// update the `buffers`, `time_span_start`, and `time_span_end` with Goldberg
  /// modal data (either Goldberg modal data or complex nodal data depending on
  /// the template parameter to this class) and the start and end index in the
  /// member `time_buffer_` covered by the newly updated `buffers`.
  double update_buffers_for_time(
      gsl::not_null<Variables<tags_for_writing>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length, size_t buffer_depth,
      bool time_varies_fastest = true) const override;

  std::unique_ptr<WorldtubeBufferUpdater<tags_for_writing>> get_clone()
      const override {
    return std::make_unique<BondiWorldtubeH5BufferUpdater>(filename_);
  }

  /// The time can only be supported in the buffer update if it is between the
  /// first and last time of the input file.
  bool time_is_outside_range(const double time) const override {
    return time < time_buffer_[0] or
           time > time_buffer_[time_buffer_.size() - 1];
  }

  /// retrieves the l_max of the input file
  size_t get_l_max() const override { return l_max_; }

  /// retrieves the extraction radius. In most normal circumstances, this will
  /// not be needed for Bondi data.
  double get_extraction_radius() const override {
    if (not extraction_radius_.has_value()) {
      ERROR(
          "Extraction radius has not been set, and was not successfully parsed "
          "from the filename. The extraction radius has been used, so must be "
          "set either by the input file or via the filename.");
    }
    return *extraction_radius_;
  }

  /// The time buffer is supplied by non-const reference to allow views to
  /// easily point into the buffer.
  ///
  /// \warning Altering this buffer outside of the constructor of this class
  /// results in undefined behavior! This should be supplied by const reference
  /// once there is a convenient method of producing a const view of a vector
  /// type.
  DataVector& get_time_buffer() override { return time_buffer_; }

  bool has_version_history() const override { return true; }

  /// Serialization for Charm++.
  void pup(PUP::er& p) override;

 private:
  std::optional<double> extraction_radius_ = std::nullopt;
  size_t l_max_ = 0;

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file_;
  std::string filename_;

  tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<Tags::detail::InputDataSet, tags_for_writing>>
      dataset_names_;

  // stores all the times in the input file
  DataVector time_buffer_;
};

/// A `WorldtubeBufferUpdater` specialized to the Klein-Gordon input worldtube
/// H5 file produced by the SpEC format. We assume the scalar field is
/// real-valued.
class KleinGordonWorldtubeH5BufferUpdater
    : public WorldtubeBufferUpdater<klein_gordon_input_tags> {
 public:
  // charm needs the empty constructor
  KleinGordonWorldtubeH5BufferUpdater() = default;

  /// The constructor takes the filename of the SpEC h5 file that will be used
  /// for boundary data. The extraction radius can either be passed in directly,
  /// or if it takes the value `std::nullopt`, then the extraction radius is
  /// retrieved as an integer in the filename.
  explicit KleinGordonWorldtubeH5BufferUpdater(
      const std::string& cce_data_filename,
      std::optional<double> extraction_radius = std::nullopt);

  WRAPPED_PUPable_decl_template(KleinGordonWorldtubeH5BufferUpdater);  // NOLINT

  explicit KleinGordonWorldtubeH5BufferUpdater(CkMigrateMessage* /*unused*/) {}

  /// update the `buffers`, `time_span_start`, and `time_span_end` with Goldberg
  /// modal data and the start and end index in the member `time_buffer_`
  /// covered by the newly updated `buffers`.
  double update_buffers_for_time(
      gsl::not_null<Variables<klein_gordon_input_tags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length, size_t buffer_depth,
      bool time_varies_fastest = true) const override;

  std::unique_ptr<WorldtubeBufferUpdater<klein_gordon_input_tags>> get_clone()
      const override {
    return std::make_unique<KleinGordonWorldtubeH5BufferUpdater>(filename_);
  }

  /// The time can only be supported in the buffer update if it is between the
  /// first and last time of the input file.
  bool time_is_outside_range(const double time) const override {
    return time < time_buffer_[0] or
           time > time_buffer_[time_buffer_.size() - 1];
  }

  /// retrieves the l_max of the input file
  size_t get_l_max() const override { return l_max_; }

  /// retrieves the extraction radius. In most normal circumstances, this will
  /// not be needed for Klein-Gordon data.
  double get_extraction_radius() const override {
    if (not static_cast<bool>(extraction_radius_)) {
      ERROR(
          "Extraction radius has not been set, and was not successfully parsed "
          "from the filename. The extraction radius has been used, so must be "
          "set either by the input file or via the filename.");
    }
    return *extraction_radius_;
  }

  /// The time buffer is supplied by non-const reference to allow views to
  /// easily point into the buffer.
  ///
  /// \warning Altering this buffer outside of the constructor of this class
  /// results in undefined behavior! This should be supplied by const reference
  /// once there is a convenient method of producing a const view of a vector
  /// type.
  DataVector& get_time_buffer() override { return time_buffer_; }

  bool has_version_history() const override { return true; }

  /// Serialization for Charm++.
  void pup(PUP::er& p) override;

 private:
  std::optional<double> extraction_radius_ = std::nullopt;
  size_t l_max_ = 0;

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file_;
  std::string filename_;

  tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<Tags::detail::InputDataSet, klein_gordon_input_tags>>
      dataset_names_;

  // stores all the times in the input file
  DataVector time_buffer_;
};
}  // namespace Cce
