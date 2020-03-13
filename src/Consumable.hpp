#ifndef __CONSUMABLE_HPP
#define __CONSUMABLE_HPP

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <thread>

template <typename T>
class Consumable {
public:

	Consumable() = default;

	Consumable(const T &obj) : _object(obj) {}

	/***
	 * check if object has been updated since last consumption
	 */
	bool available() const {
		return _available;
	}

	/***
	 * sleep until producer updates the object
	 */
	template <typename Rep, typename Period>
	void wait_for_update(const std::chrono::duration<Rep, Period> &duration=std::chrono::milliseconds(100)) const {
		while (!available()) {
			std::this_thread::sleep_for(duration);
		}
	}

	/***
	 * start consumption, until finish is called the producer cannot update
	 * the object
	 */
	T& consume() {
		_updatable = false;
		_mtx.lock();
		_available = false;
		return _object;
	}

	/***
	 * end consumption, the producer can now update the object
	 */
	void finish() {
		_mtx.unlock();
		_updatable = true;
	}

	/***
	 * update the object if it not currently consumed
	 */
	void update(const T &obj) {
		std::lock_guard<std::mutex> lock(_mtx);
		_object = obj;
		_available = true;
	}

	void update(T &&obj) {
		std::lock_guard<std::mutex> lock(_mtx);
		_object = std::forward<T>(obj);
		_available = true;
	
	
	}

	/***
	 * check if consumer is consuming the object
	 */
	bool updatable() const {
		return _updatable;
	}

	/***
	 * wait until consumer finished consumption
	 */
	template <typename Rep, typename Period>
	void wait_until_updatable(const std::chrono::duration<Rep, Period> &duration=std::chrono::milliseconds(100)) const {
		while (!updatable()) {
			std::this_thread::sleep_for(duration);
		}
	}

private:

	T _object;

	std::mutex _mtx; // protect object

	std::atomic_bool _available = { false }; // has been updated since last consumption

	std::atomic_bool _updatable = { true }; // consumer is consuming object

};

#endif // __CONSUMABLE_HPP
