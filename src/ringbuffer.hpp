#ifndef __RINGBUFFER_HPP
#define __RINGBUFFER_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

template <typename T>
class RingBuffer {
public:

	typedef std::lock_guard<std::mutex> lock_type;

	RingBuffer() = default;

	RingBuffer(size_t capacity) {
		if (capacity == 0) {
			throw std::runtime_error("cannot create empty buffer");
		}
		_begin = new T[capacity + 1];
		_end = _begin + capacity + 1;
		_head = _begin + 1;
		_tail = _begin;
	}

	RingBuffer(const RingBuffer &buffer) = delete;

	~RingBuffer() {
		if (_begin != nullptr) {
			delete[] _begin;
		}
	}

	T pop() {
		T tmp;
		pop(tmp);
		return tmp;
	}

	void pop(T &item) {
		if (empty()) {
			throw std::runtime_error("pop on buffer empty");
		}
		lock_type lock(_tmtx);
		item = *_item;
		_tail = _next(_tail);
	}

	void push(const T &item) {
		if (full()) {
			throw std::runtime_error("push on full buffer");
		}
		lock_type lock(_hmtx);
		*_head = item;
		_head = _next(_head);
	}

	void push(T &&item) {
		if (full()) {
			throw std::runtime_error("push on full buffer");
		}
		lock_type lock(_hmtx);
		*_head = std::forward(item);
		_head = _next(_head);
	}

	T peek() {
		T tmp;
		peek(tmp);
		return tmp;
	}

	void peek(T &item) {
		lock_type lock(_tmtx);
		item = *_tail;
	}

	size_t size() const {
		return _head > _tail ? _head - _tail : capacity() - (_tail - _head);
	}

	size_t capacity() const {
		return _end - _begin - 1;
	}

	bool empty() const {
		return _next(_tail) == _head;
	}

	bool full() const {
		return _head == _tail;
	}

private:

	T *_next(T *ptr) {
		ptr += 1;
		return ptr == _end ? _begin : ptr;
	}

	T *_begin = nullptr;

	T *_end = nullptr;

	T *_head = nullptr;

	T *_tail = nullptr;

	std::mutex _hmtx;

	std::mutex _tmtx;

};

#endif // __RINGBUFFER_HPP
