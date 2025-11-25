// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>  // for std::memcpy
#include <new>      // for hardware_destructive_interference_size
#include <optional>
#include <ostream>
#include <type_traits>
#include <vector>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ForceInline.hpp"

namespace Parallel {
namespace lockfree {
/*!
 * \brief A lockfree multi-producer multi-consumer unordered set for types
 * that are at most 8 bytes in size.
 *
 * In order to provide fully lockfree semantics, the number of buckets must
 * chosen at construction. If the capacity (roughly number of buckets times size
 * of each bucket, but this depends on how the objects hash into the set) is
 * reached, no more keys can be inserted. For improved performance, the number
 * of buckets must be a positive power of 2, i.e. `2^N` for `N>-1`.
 *
 * Performance degradation should be expected at 50% capacity at the earliest,
 * though may well be much higher due to an extremely cache-friendly
 * implementation.
 *
 * Users can optionally specify the sentinel value used to mark a slot as
 * empty by specifying the `EmptySlotValue`. By default the
 * `EmptySlotValue` is 0. It is undefined behavior if a key with the
 * value of the `EmptySlotValue` is inserted. No diagnostic is provided.
 *
 * Finally, users can choose between two different implementations, one which
 * tracks the bucket sizes and one which does not. We have not compared
 * performance, though the non-tracking may perform better since no memory
 * synchronizations are necessary and since each bucket fits into one cache
 * line. Thus, each load from memory loads not just the bucket but all entries
 * in the bucket, reducing the cost of linear probing inside a bucket.
 *
 * \warning This class does not synchronize memory, which means while all
 * operations on the container is atomic, it cannot be used to synchronize
 * data not contained in the set across different cores.
 *
 * ### Implementation details
 *
 * The set uses a single contiguous memory layout aligned on
 * `std::hardware_destructive_interference_size`, typically 64 bytes on 2025
 * hardware. Since each entry in the set is a `std::uint64_t`, we choose each
 * bucket to be of size `std::hardware_destructive_interference_size/8`, so 8
 * on 2025 hardware. When a key is inserted, it is first hashed using a
 * Fibonacci multiplicative hash to create a uniform distribution over the
 * buckets. Then, if `TrackBucketSize` is false it is inserted using
 * `memory_order_relaxed` in the first empty slot in the bucket. If
 * `TrackBucketSize` is true then it is inserted in the first empty slot that
 * is not slot 0. Slot 0 is used to track the last index in the slot and is
 * increased by 1 on insert if the last index is less than the bucket size
 * (note: the bucket size is 1 smaller when tracking the size since the size
 * is part of the bucket). The insert into the slot is done using
 * `memory_order_relaxed` and the bucket size increase is done using
 * `memory_order_release`.
 *
 * Since the buckets are cache aligned and the size of one cache line,
 * performance is expected to remain strong even for very high load factors.
 */
template <class T, std::uint64_t EmptySlotValue = 0,
          bool TrackBucketSize = false>
class FixedSizeUnorderedSet {
 private:
#ifdef __cpp_lib_hardware_interference_size
  static constexpr size_t cache_line_size_ =
      std::hardware_destructive_interference_size;
#else
  static constexpr size_t cache_line_size_ = 64;
#endif

 public:
  static_assert(sizeof(T) <= sizeof(std::uint64_t));
  static_assert(cache_line_size_ > sizeof(std::uint64_t));
  static_assert(cache_line_size_ % sizeof(std::uint64_t) == 0);

  /// \brief The number of elements in each bucket.
  ///
  /// This depends on the hardware, but on circa 2025 CPUs the cacheline size
  /// is 64 bytes.
  static constexpr std::uint64_t bucket_size =
      cache_line_size_ / sizeof(std::uint64_t);

  /// \brief The default empty slot value.
  static constexpr std::uint64_t default_empty_slot_value = 0;
  /// \brief The empty slot value.
  static constexpr std::uint64_t empty_slot_value = EmptySlotValue;

  /*!
   * \brief Create a multi-producer multi-consumer unordered set that allows
   * at most `number_of_buckets * bucket_size` objects.
   *
   * \warning \p number_of_buckets must be a power of two greater than 0.
   */
  explicit FixedSizeUnorderedSet(size_t number_of_buckets);
  // Delete copy and move constructors and assignment operators since this
  // class stores atomic variables needed for thread-safety.
  FixedSizeUnorderedSet(const FixedSizeUnorderedSet&) = delete;
  FixedSizeUnorderedSet& operator=(const FixedSizeUnorderedSet&) = delete;
  FixedSizeUnorderedSet(FixedSizeUnorderedSet&&) = delete;
  FixedSizeUnorderedSet& operator=(FixedSizeUnorderedSet&&) = delete;
  ~FixedSizeUnorderedSet() = default;

  /*!
   * \brief Insert the \p key into the set.
   *
   * \param key The key to insert.
   * \return `true` if the key was inserted or found in the set. Returns
   *         `false` if we failed to insert the key because we reached
   *         \p max_linear_probes.
   */
  [[nodiscard]] bool insert(T key) noexcept;

  /*!
   * \brief Erase the \p key from the set.
   *
   * \param key The key to erase.
   * \return `true` if the key was erased by this thread. Returns `false` if we
   *         failed to erase the key because we reached \p max_linear_probes
   *         or because another thread erased the key. We have no general way of
   *         knowing why we could not find the \p key to erase.
   */
  [[nodiscard]] bool erase(T key) noexcept;

  /*!
   * \brief Check if the unordered set contains the \p key.
   *
   * \param key The key to check if it is contained in the unordered set.
   * \return `true` if the \p key is found, `false` if not.
   *
   * \warning A \p key may be found in the unordered set and then immediately
   * erased by another thread.
   */
  [[nodiscard]] bool contains(T key) const noexcept;

  /// \brief Returns the number of buckets.
  [[nodiscard]] constexpr size_t number_of_buckets() const noexcept {
    return number_of_buckets_;
  }

  /// \brief The maximum number of entries.
  ///
  /// You cannot have more entries than this, but it is only possible to reach
  /// this number of entries if all chains are used, which is extremely
  /// unlikely. In practice, a reasonable capacity is around 0.5-0.8 times the
  /// number of buckets.
  [[nodiscard]] constexpr size_t maximum_capacity() const noexcept {
    return number_of_buckets() * (bucket_size - (TrackBucketSize ? 1 : 0));
  }

  /// \brief Returns the (approximate) size of the set.
  ///
  /// The exact size is not well-defined since this is a concurrent data
  /// structure.
  [[nodiscard]] std::uint64_t approximate_size() const noexcept {
    return size_.load(std::memory_order_relaxed);
  }

  /// \brief The (approximate) load factor
  ///
  /// The exact load factor is not well-defined since this is a concurrent data
  /// structure.
  [[nodiscard]] constexpr double approximate_load_factor() const noexcept {
    return static_cast<double>(approximate_size()) /
           static_cast<double>(number_of_buckets());
  }

 private:
  [[nodiscard]] constexpr SPECTRE_ALWAYS_INLINE std::uint64_t
  compute_internal_key(const T key) const {
    std::uint64_t result{0};
    std::memcpy(&result, &key, sizeof(T));
    return result;
  }

  // Implements a Fibonacci (multiplicative) hash. This gives an integer in
  // [0,2^d) which is used to index the hash table. Since we have a list
  // length of cache_line_size_/sizeof(uint64_t)~8, it is unlikely that even
  // if we reach a load factor of 1 we will be worse than a few linear probes,
  // so still O(1+8)~O(1).
  [[nodiscard]] constexpr SPECTRE_ALWAYS_INLINE std::uint64_t hash(
      const std::uint64_t index) const noexcept {
    // z ~ 2^64/1.6180339, we divide by the Golden Ratio
    constexpr std::uint64_t z{11400714819323198485LLU};
    constexpr std::uint64_t w = 64;
    return (z * index) >> (w - dimension_);
  }

  // Since atomics may not always be lock free depending on alignment, we want
  // to catch issues where the hardware cannot guarantee that the atomic is
  // lock free. It is not inherently bad that we cannot guarantee that the
  // atomics are lock free at compile time, we could check at runtime, but
  // it's nice to have the guarantee when possible. If for some reason the
  // guarantee isn't given, then likely forcing alignment of the individual
  // atomic variables would restore it. For example, some system may only be
  // able to handle atomics on 32-byte word boundaries.
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

  struct alignas(cache_line_size_) AlignedEntry {
    std::atomic<std::uint64_t> value{0};
  };

  alignas(cache_line_size_)
      std::unique_ptr<
          // NOLINTNEXTLINE(modernize-avoid-c-arrays)
          std::atomic<std::uint64_t>[], decltype([](void* ptr) {
            operator delete[](
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<std::atomic<std::uint64_t>*>(ptr),
                std::align_val_t{cache_line_size_});
          })> entries_ {};
  std::uint64_t number_of_buckets_{0};
  std::uint64_t dimension_{0};
  alignas(cache_line_size_) std::atomic<std::uint64_t> size_{0};

  // Ensure we pad the end to avoid false sharing. Unlikely to happen because
  // of the layout, but better to be safe.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  std::byte padding_[cache_line_size_ - 2 * sizeof(std::uint64_t) -
                     sizeof(decltype(entries_))] = {};
};

template <class T, std::uint64_t EmptySlotValue, bool TrackBucketSize>
FixedSizeUnorderedSet<T, EmptySlotValue, TrackBucketSize>::
    FixedSizeUnorderedSet(const size_t number_of_buckets)
    : entries_(new(std::align_val_t{cache_line_size_})
                   std::atomic<std::uint64_t>[number_of_buckets * bucket_size]),
      number_of_buckets_(number_of_buckets) {
  if (number_of_buckets_ == 0) {
    ERROR("The capacity must be a power of two larger than 0. Got "
          << number_of_buckets_);
  }
  if ((number_of_buckets_ bitand (number_of_buckets_ - 1)) != 0) {
    ERROR("The capacity must be a power of two larger than 0. Got "
          << number_of_buckets_);
  }
  const auto uint64_log2 = [](uint64_t n) -> std::uint64_t {
    int i = -static_cast<int>(n == 0);

    const auto helper = [&i, &n](auto k) -> void {
      if (n >= (static_cast<std::uint64_t>(1) << decltype(k)::value)) {
        i += static_cast<int>(decltype(k)::value);
        n >>= decltype(k)::value;
      }
    };
    helper(std::integral_constant<size_t, 32>{});
    helper(std::integral_constant<size_t, 16>{});
    helper(std::integral_constant<size_t, 8>{});
    helper(std::integral_constant<size_t, 4>{});
    helper(std::integral_constant<size_t, 2>{});
    helper(std::integral_constant<size_t, 1>{});
    return static_cast<std::uint64_t>(i);
  };
  dimension_ = uint64_log2(number_of_buckets_);

  // Explicitly zero out the counters.
  for (size_t i = 0; i < number_of_buckets_ * bucket_size; ++i) {
    entries_[i].store(EmptySlotValue, std::memory_order_relaxed);
  }
}

template <class T, std::uint64_t EmptySlotValue, bool TrackBucketSize>
bool FixedSizeUnorderedSet<T, EmptySlotValue, TrackBucketSize>::insert(
    const T key) noexcept {
  const std::uint64_t internal_key = compute_internal_key(key);
  const std::uint64_t bucket_index = hash(internal_key) * bucket_size;
  if constexpr (TrackBucketSize) {
    std::atomic<std::uint64_t>& this_bucket_size = entries_[bucket_index];
    // We first do a linear search in the bucket to see if the element exists.
    for (size_t i = 0; i < this_bucket_size.load(std::memory_order_relaxed);
         ++i) {
      const size_t index = bucket_index + 1 + i;
      const std::uint64_t probed_internal_key =
          entries_[index].load(std::memory_order_relaxed);
      if (probed_internal_key == internal_key) {
        return true;  // set already contains value
      }
    }

    // We now know that _if_ the element exists, it's in what was in the first
    // empty slot, which means we either encounter the element in the first
    // available slot or we get to insert it in that slot.
    for (size_t i = this_bucket_size.load(std::memory_order_relaxed);
         i < (bucket_size - 1); ++i) {
      const size_t index = bucket_index + 1 + i;
      const std::uint64_t probed_internal_key =
          entries_[index].load(std::memory_order_relaxed);
      if (probed_internal_key == internal_key) {
        return true;  // set already contains value
      } else {
        if (probed_internal_key != EmptySlotValue) {
          // The entry is used by another key.
          continue;
        }
        // The entry is empty. Let's try to set it.
        std::uint64_t current_key_in_slot = EmptySlotValue;
        if (not entries_[index].compare_exchange_strong(
                current_key_in_slot, internal_key, std::memory_order_relaxed,
                std::memory_order_relaxed) and
            current_key_in_slot != internal_key) {
          // Another thread just stole this slot from us. Try next slot.
          continue;
        }
        size_.fetch_add(1, std::memory_order_relaxed);
        if (this_bucket_size.load(std::memory_order_relaxed) <
            bucket_size - 1) {
          this_bucket_size.fetch_add(1, std::memory_order_release);
        }
        // Successful insert. Return.
        return true;
      }
    }
  } else {
    // We first do a linear search in the bucket to see if the element exists.
    for (size_t i = 0; i < bucket_size; ++i) {
      const size_t index = bucket_index + i;
      const std::uint64_t probed_internal_key =
          entries_[index].load(std::memory_order_relaxed);
      if (probed_internal_key == internal_key) {
        return true;  // set already contains value
      }
    }

    // We now know that _if_ the element exists, it's in what was in the first
    // empty slot, which means we either encounter the element in the first
    // available slot or we get to insert it in that slot.
    for (size_t i = 0; i < bucket_size; ++i) {
      const size_t index = bucket_index + i;
      const std::uint64_t probed_internal_key =
          entries_[index].load(std::memory_order_relaxed);
      if (probed_internal_key != EmptySlotValue) {
        // The entry is used by another key.
        continue;
      }
      // The entry is empty. Let's try to set it.
      std::uint64_t current_key_in_slot = EmptySlotValue;
      if (not entries_[index].compare_exchange_strong(
              current_key_in_slot, internal_key, std::memory_order_relaxed,
              std::memory_order_relaxed) and
          current_key_in_slot != internal_key) {
        // Another thread just stole this slot from us. Try next slot.
        continue;
      }
      size_.fetch_add(1, std::memory_order_relaxed);
      // Successful insert. Return.
      return true;
    }
  }
  return false;
}

template <class T, std::uint64_t EmptySlotValue, bool TrackBucketSize>
bool FixedSizeUnorderedSet<T, EmptySlotValue, TrackBucketSize>::erase(
    const T key) noexcept {
  const std::uint64_t internal_key = compute_internal_key(key);
  const std::uint64_t bucket_index = hash(internal_key) * bucket_size;
  if constexpr (TrackBucketSize) {
    std::atomic<std::uint64_t>& this_bucket_size = entries_[bucket_index];
    for (size_t i = 0; i < this_bucket_size.load(std::memory_order_relaxed);
         ++i) {
      const size_t index = bucket_index + 1 + i;
      const std::uint64_t probed_internal_key =
          entries_[index].load(std::memory_order_relaxed);
      if (probed_internal_key == internal_key) {
        std::uint64_t current_key_in_slot = internal_key;
        if (not entries_[index].compare_exchange_strong(
                current_key_in_slot, EmptySlotValue, std::memory_order_relaxed,
                std::memory_order_relaxed) and
            current_key_in_slot != EmptySlotValue) {
          // If the CAS failed and the slot value is not the EmptySlotValue,
          // then we failed to erase. E.g. another thread could have erased this
          // value.
          // NOLINTNEXTLINE(readability-simplify-boolean-expr)
          return false;
        }
        size_.fetch_sub(1, std::memory_order_relaxed);
        // Shrinking the bucket size is extremely difficult because we use the
        // bucket size as the index of one past the last element in the
        // bucket, which may be larger than the number of elements in the
        // bucket. This means we can only shrink the size if we are removing
        // the last element, but knowing we erased the last element is
        // difficult in parallel.
        return true;
      }
    }
  } else {
    for (size_t i = 0; i < bucket_size; ++i) {
      const size_t index = bucket_index + i;
      const std::uint64_t probed_internal_key =
          entries_[index].load(std::memory_order_relaxed);
      if (probed_internal_key == internal_key) {
        std::uint64_t current_key_in_slot = internal_key;
        if (not entries_[index].compare_exchange_strong(
                current_key_in_slot, EmptySlotValue, std::memory_order_relaxed,
                std::memory_order_relaxed) and
            current_key_in_slot != EmptySlotValue) {
          // If the CAS failed and the slot value is not the EmptySlotValue,
          // then we failed to erase. E.g. another thread could have erased this
          // value.
          // NOLINTNEXTLINE(readability-simplify-boolean-expr)
          return false;
        }
        size_.fetch_sub(1, std::memory_order_relaxed);
        return true;
      }
    }
  }
  return false;
}

template <class T, std::uint64_t EmptySlotValue, bool TrackBucketSize>
bool FixedSizeUnorderedSet<T, EmptySlotValue, TrackBucketSize>::contains(
    const T key) const noexcept {
  const std::uint64_t internal_key = compute_internal_key(key);
  const std::uint64_t bucket_index = hash(internal_key) * bucket_size;
  if constexpr (TrackBucketSize) {
    const std::atomic<std::uint64_t>& this_bucket_size = entries_[bucket_index];
    for (size_t i = 0; i < this_bucket_size.load(std::memory_order_relaxed);
         ++i) {
      const size_t index = bucket_index + 1 + i;
      const std::uint64_t probed_internal_key =
          entries_[index].load(std::memory_order_relaxed);
      if (probed_internal_key == internal_key) {
        return true;
      }
    }
  } else {
    for (size_t i = 0; i < bucket_size; ++i) {
      const size_t index = bucket_index + i;
      const std::uint64_t probed_internal_key =
          entries_[index].load(std::memory_order_relaxed);
      if (probed_internal_key == internal_key) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace lockfree
}  // namespace Parallel
