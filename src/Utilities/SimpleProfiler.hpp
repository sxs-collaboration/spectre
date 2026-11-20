// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// Things related to basic profiling.
namespace profiling {
/// A type used by `SimpleProfiler` labels to disable runtime labeling.
struct NoLabel {
  friend bool operator==(const NoLabel&, const NoLabel&) = default;
};

constexpr size_t hash_value(const NoLabel& /*unused*/) { return 0; }

template <typename Label>
concept ProfilerLabelWithoutColumns = requires(Label::FileLabel file_label) {
  requires std::is_trivially_copyable_v<typename Label::FileLabel>;
  {
    boost::hash<typename Label::FileLabel>{}(file_label)
  } -> std::same_as<size_t>;
  { Label::make_file_name(file_label) } -> std::convertible_to<std::string>;

  requires std::same_as<typename Label::ColumnLabel, NoLabel>;
};

template <typename Label>
concept ProfilerLabelWithColumns =
    requires(Label::FileLabel file_label, Label::ColumnLabel column_label) {
      requires std::is_trivially_copyable_v<typename Label::FileLabel>;
      {
        boost::hash<typename Label::FileLabel>{}(file_label)
      } -> std::same_as<size_t>;
      { Label::make_file_name(file_label) } -> std::convertible_to<std::string>;

      requires std::is_trivially_copyable_v<typename Label::ColumnLabel>;
      {
        boost::hash<typename Label::ColumnLabel>{}(column_label)
      } -> std::same_as<size_t>;
      requires not std::same_as<typename Label::ColumnLabel, NoLabel>;

      {
        Label::make_column_name(column_label)
      } -> std::convertible_to<std::string>;
      requires std::same_as<
          std::remove_cvref_t<decltype(*begin(Label::allowed_column_labels))>,
          typename Label::ColumnLabel>;
    };

/// Concept for a class usable as the label for a `SimpleProfiler`
///
/// All implementations must provide type aliases `FileLabel` and
/// `ColumnLabel`, which can be used to distinguish between profiled
/// data at runtime.  The `FileLabel` can be used to specify the
/// output subfile for the profiler data, and the `ColumnLabel`
/// distinguishes columns in a subfile.  Both types must be trivially
/// copyable and hashable using `boost::hash`.  The implementation
/// assumes there will be no hash collisions, and timing results may
/// be mixed unpredictably if collisions occur.  If a particular label
/// does not wish to use one or both of these runtime options, it
/// should set them to `NoLabel`.
///
/// Implementations must provide a function `make_file_name` to
/// translate the `FileLabel` objects to subfile names.  File labels
/// (either for the same or different `Label` object) that produce the
/// same string are considered equivalent.  All equivalent labels must
/// produce the same set of column labels.
///
/// Implementations that do not set `ColumnLabel` to `NoLabel` must
/// provide a `make_column_name` function analogous to
/// `make_file_name`, with the same requirements.  As the columns in
/// an H5 subfile must be fixed, it must also provide a list of all
/// `allowed_column_labels`.
///
/// Examples:
///
/// \snippet Test_SimpleProfiler.cpp labels
template <typename Label>
concept ProfilerLabel =
    ProfilerLabelWithoutColumns<Label> or ProfilerLabelWithColumns<Label>;

class SimpleProfilerBase {
 public:
  void shadow(std::uint64_t nsec);

 protected:
  SimpleProfilerBase() = default;
  ~SimpleProfilerBase() = default;

  std::uint64_t shadowed_nsec_ = 0;
};

/// The data storage for `SimpleProfiler` objects.
///
/// Records lifetimes of `SimpleProfiler` objects, as well as the
/// unprofiled time on the thread.  Profilers associated with the same
/// `SimpleProfilerData` instance must profile either disjoint or
/// nested intervals.  Nested time will be recorded as exclusive time
/// on the innermost profiler and inclusive time on all active
/// profilers.
///
/// This effectively limits a single `SimpleProfilerData` object to be
/// usable only on one thread.  The data readout can be performed from
/// a different thread, and the structures are locked internally to
/// make this safe.
struct alignas(64) SimpleProfilerData {
  // Aligned to 64 bytes to avoid false sharing.

  /// Recorded number of calls and exclusive and inclusive times for a
  /// set of labels in nanoseconds.
  struct Times {
    std::uint64_t call_count = 0;
    std::uint64_t exclusive_nsec = 0;
    std::uint64_t inclusive_nsec = 0;

    void pup(PUP::er& p);
  };

  SimpleProfilerData();
  ~SimpleProfilerData() = default;
  SimpleProfilerData(const SimpleProfilerData&) = delete;
  SimpleProfilerData& operator=(const SimpleProfilerData&) = delete;

  SimpleProfilerData(SimpleProfilerData&& other) noexcept;

  SimpleProfilerData& operator=(SimpleProfilerData&& other) noexcept;

  // We use an ordered map for the inner container for the convenience
  // of the observer, which wants the columns in a well-defined order
  // but doesn't care about the order of the subfiles.
  using TimingData =
      std::unordered_map<std::string, std::map<std::string, Times>>;

  /// Obtain a copy of the timing data.
  ///
  /// This returns a copy because accessing the internal data
  /// structures is not thread-safe.
  ///
  /// The return type is a map of maps to `Times` objects.  The outer
  /// map is indexed by the "file" names provided by the profiler
  /// labels and the inner map by the "column" names.  Profiler labels
  /// that do not provide column names result in a single entry in the
  /// inner map with an empty string as the key.
  TimingData timing_data();

  void pup(PUP::er& p);

  // Functions called by SimpleProfiler objects.
 private:
  template <typename>
  friend class SimpleProfiler;

  using ProfilerState = std::pair<timespec, SimpleProfilerBase*>;

  ProfilerState start_profiling(gsl::not_null<SimpleProfilerBase*> profiler);

  template <typename Label>
  void end_profiling(const gsl::not_null<SimpleProfilerBase*> profiler,
                     const Label::FileLabel& file_label,
                     const Label::ColumnLabel& column_label,
                     const ProfilerState& state,
                     const std::uint64_t shadowed_nsec) {
    const auto end_time = current_time();
    const std::scoped_lock lock(mutex_);
    ASSERT(profiler == active_profiler_, "Ending profiling for wrong profiler");
    const auto inclusive_nsec = record_result<Label>(
        file_label, column_label, state.first, end_time, shadowed_nsec);
    if (state.second != nullptr) {
      state.second->shadow(inclusive_nsec);
    }
    active_profiler_ = state.second;
  }

  void active_profiler_is_moving(gsl::not_null<SimpleProfilerBase*> source,
                                 gsl::not_null<SimpleProfilerBase*> dest);

  // Actual internal-use helpers
 private:
  static timespec current_time();

  void update_unprofiled_time(const timespec& now);

  template <typename Label>
  std::uint64_t record_result(const Label::FileLabel& file_label,
                              const Label::ColumnLabel& column_label,
                              const timespec& start_time,
                              const timespec& end_time,
                              const std::uint64_t shadowed_nsec) {
    const std::uint64_t inclusive_nsec =
        static_cast<std::uint64_t>(end_time.tv_sec - start_time.tv_sec) *
            1'000'000'000u +
        static_cast<std::uint64_t>(end_time.tv_nsec) -
        static_cast<std::uint64_t>(start_time.tv_nsec);
    const std::uint64_t exclusive_nsec = inclusive_nsec - shadowed_nsec;

    const std::tuple<std::type_index, size_t, size_t> key{
        typeid(Label), boost::hash<typename Label::FileLabel>{}(file_label),
        boost::hash<typename Label::ColumnLabel>{}(column_label)};
    auto entry = type_times_.find(key);
    if (UNLIKELY(entry == type_times_.end())) {
      const auto file_name_entry =
          times_.try_emplace(Label::make_file_name(file_label)).first;
      if constexpr (ProfilerLabelWithoutColumns<Label>) {
        const auto column_name_entry =
            file_name_entry->second.try_emplace("").first;
        entry = type_times_.try_emplace(key, &column_name_entry->second).first;
      } else {
        for (const auto& allowed_column_label : Label::allowed_column_labels) {
          const auto column_name_entry =
              file_name_entry->second
                  .try_emplace(Label::make_column_name(allowed_column_label))
                  .first;
          const std::tuple<std::type_index, size_t, size_t> col_key{
              get<0>(key), get<1>(key),
              boost::hash<typename Label::ColumnLabel>{}(allowed_column_label)};
          const auto type_entry =
              type_times_.try_emplace(col_key, &column_name_entry->second)
                  .first;
          if (allowed_column_label == column_label) {
            entry = type_entry;
          }
        }
        ASSERT(entry != type_times_.end(),
               "Column label not in allowed_column_labels");
      }
    }
    ++entry->second->call_count;
    entry->second->exclusive_nsec += exclusive_nsec;
    entry->second->inclusive_nsec += inclusive_nsec;

    updated_time_ = end_time;

    return inclusive_nsec;
  }

  std::mutex mutex_{};
  SimpleProfilerBase* active_profiler_ = nullptr;
  timespec updated_time_ = current_time();

  TimingData times_;
  // Pointers into times_ keyed by a structure that is fast to
  // compute.  The internal type_index keys cannot be serialized, so
  // this map is discarded and rebuilt after deserialization.
  std::unordered_map<std::tuple<std::type_index, size_t, size_t>,
                     gsl::not_null<Times*>,
                     boost::hash<std::tuple<std::type_index, size_t, size_t>>>
      type_times_;
};

/// A simple scope-based profiler
///
/// Records the time the `SimpleProfiler` object is alive in the
/// associated `SimpleProfilerData` object.  The `Label` template
/// parameter must satisfy the `ProfilerLabel` concept.
template <typename Label>
class [[nodiscard]] SimpleProfiler final : public SimpleProfilerBase {
 public:
  /// Start a profiling region, with initial values for the file and
  /// column labels.  Labels of type `profiling::NoLabel` may be
  /// omitted.  The \p profiler_data may be null, in which case no
  /// timing is performed.
  SimpleProfiler(SimpleProfilerData* const profiler_data,
                 const Label::FileLabel& file_label,
                 const Label::ColumnLabel& column_label)
    requires ProfilerLabel<Label>
      : profiler_data_(profiler_data),
        file_label_(file_label),
        column_label_(column_label) {
    if (LIKELY(profiler_data_ != nullptr)) {
      state_ = profiler_data_->start_profiling(this);
    }
  }

  SimpleProfiler(SimpleProfilerData* const profiler_data,
                 const Label::ColumnLabel& column_label)
    requires std::same_as<typename Label::FileLabel, NoLabel>
      : SimpleProfiler(profiler_data, NoLabel{}, column_label) {}

  SimpleProfiler(SimpleProfilerData* const profiler_data,
                 const Label::FileLabel& file_label)
    requires std::same_as<typename Label::ColumnLabel, NoLabel>
      : SimpleProfiler(profiler_data, file_label, NoLabel{}) {}

  explicit SimpleProfiler(SimpleProfilerData* const profiler_data)
    requires std::same_as<typename Label::FileLabel, NoLabel> and
             std::same_as<typename Label::ColumnLabel, NoLabel>
      : SimpleProfiler(profiler_data, NoLabel{}, NoLabel{}) {}

  ~SimpleProfiler() { end_profiling(); }

  /// Move construction is allowed so consumers can write factory
  /// functions if they wish.  Move assignment would usually cause the
  /// construction and destruction orders to interleave improperly, so
  /// is not allowed.
  SimpleProfiler(SimpleProfiler&& other) noexcept
      : SimpleProfilerBase(std::move(other)),
        profiler_data_(other.profiler_data_),
        state_(std::move(other.state_)),
        file_label_(other.file_label_),
        column_label_(other.column_label_) {
    other.profiler_data_ = nullptr;
    if (LIKELY(profiler_data_ != nullptr)) {
      profiler_data_->active_profiler_is_moving(&other, this);
    }
  }

  SimpleProfiler(const SimpleProfiler&) = delete;
  SimpleProfiler& operator=(const SimpleProfiler&) = delete;
  SimpleProfiler& operator=(SimpleProfiler&&) = delete;

  /// Change the file label the time will be recorded under.  This
  /// change is retroactive and only the value when the profiled
  /// region ends matters.
  void set_file_label(const Label::FileLabel& file_label) {
    file_label_ = file_label;
  }

  /// Change the column label the time will be recorded under.  This
  /// change is retroactive and only the value when the profiled
  /// region ends matters.
  void set_column_label(const Label::ColumnLabel& column_label) {
    column_label_ = column_label;
  }

  /// Stop profiling the current region.  This is automatically done
  /// when the profiler goes out of scope, but can be performed
  /// manually when scoping is inconvenient.  A stopped profiler
  /// cannot be restarted.
  void end_profiling() {
    if (LIKELY(profiler_data_ != nullptr)) {
      profiler_data_->end_profiling<Label>(this, file_label_, column_label_,
                                           state_, shadowed_nsec_);
      profiler_data_ = nullptr;
    }
  }

 private:
  SimpleProfilerData* profiler_data_ = nullptr;
  SimpleProfilerData::ProfilerState state_{};
  Label::FileLabel file_label_{};
  Label::ColumnLabel column_label_{};
};
}  // namespace profiling
