// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <atomic>
#include <chrono>
#include <thread>

#include "Parallel/MultiReaderSpinlock.hpp"

SPECTRE_TEST_CASE("Unit.Parallel.MultiReaderSpinlock", "[Parallel][Unit]") {
  // It's very difficult to test a lock since they are designed to prevent race
  // conditions (undefined behavior). We try to do the following just as a
  // basic sanity check that attempts to make sure things "work"

  Parallel::MultiReaderSpinlock mrsl{};
  {
    // Test two readers can lock at the same time
    std::thread t0{[&mrsl]() {
      mrsl.read_lock();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      mrsl.read_unlock();
    }};
    std::thread t1{[&mrsl]() {
      const auto start = std::chrono::high_resolution_clock::now();
      mrsl.read_lock();
      const auto end = std::chrono::high_resolution_clock::now();
      mrsl.read_unlock();
      const std::chrono::duration<long, std::nano> diff{end - start};
      // Verify it took less than 1 millisecond to lock and unlock the readlock.
      // This is a lot less than the 10ms we sleep the other thread for, even in
      // a debug build, and the lock should be acquired in about 1us in a debug
      // build.
      CHECK(diff.count() < 1000000);
    }};
    t0.join();
    t1.join();
  }
  {
    // Test that if a reader has a read-lock the writer cannot acquire a
    // write-lock.
    std::atomic<bool> locked{false};
    std::thread t0{[&mrsl, &locked]() {
      mrsl.read_lock();
      locked.store(true, std::memory_order_release);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      mrsl.read_unlock();
    }};
    while (not locked.load(std::memory_order_acquire)) {
    }
    std::thread t1{[&mrsl]() {
      const auto start = std::chrono::high_resolution_clock::now();
      mrsl.write_lock();
      const auto end = std::chrono::high_resolution_clock::now();
      mrsl.write_unlock();
      const std::chrono::duration<long, std::nano> diff{end - start};
      // Verify it took more than 1ms to retrieve the lock.
      CHECK(diff.count() > 1000000);
    }};
    t0.join();
    t1.join();
  }
  {
    // Test that if a writer has a write-lock another writer cannot acquire a
    // write-lock.
    std::atomic<bool> locked{false};
    std::thread t0{[&mrsl, &locked]() {
      mrsl.write_lock();
      locked.store(true, std::memory_order_release);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      mrsl.write_unlock();
    }};
    while (not locked.load(std::memory_order_acquire)) {
    }
    std::thread t1{[&mrsl]() {
      const auto start = std::chrono::high_resolution_clock::now();
      mrsl.write_lock();
      const auto end = std::chrono::high_resolution_clock::now();
      mrsl.write_unlock();
      const std::chrono::duration<long, std::nano> diff{end - start};
      // Verify it took more than 1ms to retrieve the lock.
      CHECK(diff.count() > 1000000);
    }};
    t0.join();
    t1.join();
  }
  {
    // Test that if a writer has a write-lock reader cannot acquire a
    // read-lock.
    std::atomic<bool> locked{false};
    std::thread t0{[&mrsl, &locked]() {
      mrsl.write_lock();
      locked.store(true, std::memory_order_release);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      mrsl.write_unlock();
    }};
    while (not locked.load(std::memory_order_acquire)) {
    }
    std::thread t1{[&mrsl]() {
      const auto start = std::chrono::high_resolution_clock::now();
      mrsl.read_lock();
      const auto end = std::chrono::high_resolution_clock::now();
      mrsl.read_unlock();
      const std::chrono::duration<long, std::nano> diff{end - start};
      // Verify it took more than 1ms to retrieve the lock.
      CHECK(diff.count() > 1000000);
    }};
    t0.join();
    t1.join();
  }
}
