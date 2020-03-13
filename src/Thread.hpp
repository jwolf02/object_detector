#ifndef __THREAD_HPP
#define __THREAD_HPP

#include <thread>
#include <atomic>
#include <stdexcept>
#include <functional>

class Thread {
public:

	Thread() = default;

	Thread(const Thread &t) = delete;

	Thread(Thread &&t) noexcept {
		swap(t);
	}

	virtual ~Thread() {
		terminate();
	}

	Thread& operator=(const Thread &t) = delete;

	Thread& operator=(Thread &&t) noexcept {
		swap(t);
		return *this;
	}

	void run() {
		if (running()) {
			throw std::runtime_error("thread is already running");
		}
		setup();
		_running = true;
		_thread = std::thread([&](){ while (_running) loop(); });
	}

	void terminate() {
		if (running()) {
			_running = false;
			_thread.join();
			cleanup();
		}
	}

	bool running() const {
		return _running;
	}

	virtual void setup() = 0;

	virtual void loop() = 0;

	virtual void cleanup() = 0;

	void swap(Thread &t) {
		std::swap(_thread, t._thread);
		_running = t._running.exchange(_running);
	}

private:

	std::thread _thread;

	std::atomic_bool _running = { false };

};

/*

class RunThread : public Thread {
public:

    static const std::function<void (void)> NONE = [](){};
    
    template <typename Setup, typename Loop, typename Cleanup>
    RunThread(const Setup &setup_func, const Loop loop_func, const Cleanup &cleanup_func) {
        _setup = [&](){ setup_func(); };
        _loop = [&](){ loop_func(); };
        _cleanup = [&](){ _cleanup_func(); };
    }

    virtual void setup() {
        _setup();
    }

    virtual void loop() {
        _loop();
    }

    virtual void cleanup() {
        _cleanup();
    }

private:

    std::function<void, (void)> _setup;

    std::function<void. (void)> _loop;

    std::function<void, (void)> _cleanup;

}:

*/

#endif // __THREAD_HPP
