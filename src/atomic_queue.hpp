#ifndef __ATOMIC_QUEUE_HPP
#define __ATOMIC_QUEUE_HPP

#include <thread>
#include <mutex>
#include <atomic>
#include <stdexcept>

template <typename T>
class atomic_queue {
public:

	struct iterator {
		iterator(const T *begin
	};

	typedef std::lock_guard<std::mutex> lock_type;

	atomic_queue() = default;

	atomic_queue(size_t size);

	~atomic_queue();

	T pop();

	void pop(T &item);

	void push(const T &item);

	void push(T &&item);

	T peek() const {
		T tmp;
		peek(tmp);
		return tmp;
	}

	void peek(T &item) {
		if (empty()) {
			throw std::runtime_error("empty buffer");
		}
		std::lock_guard<std::mutex> lock(_rmtx);
		item = *_head;
	}

	iterator begin() const;

	iterator end() const;

	size_t size() const {
		lock_type lock(_smtx);
		return _end - _begin;
	}

	bool empty() const {
		return size() == 0;
	}

	size_t max_size() const;

private:

	T *_begin = nullptr;

	T *_end = nullptr;

	T *_head = nullptr;

	T *_tail = nullptr;

	std::mutex _rmtx;

	std::mutex _wmtx;

	std::mutex _smtx;

};

#endif // __ATOMIC_QUEUE_HPP
