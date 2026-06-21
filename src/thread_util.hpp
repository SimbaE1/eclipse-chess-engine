#pragma once

#include <cstddef>
#include <functional>
#include <pthread.h>
#include <utility>

namespace eclipse {

// A joinable thread with a large stack, move-only and API-compatible with the
// subset of std::thread we use (construct-from-callable, joinable(), join()).
//
// Why this exists: the default secondary-thread stack on macOS is only 512 KB.
// The recursive alpha-beta search (negamax + qsearch, with check extensions that
// keep depth constant while in check) can drive recursion to ~150+ plies on
// sharp, forcing positions once the transposition table is warm mid-game. At a
// few KB per frame that overflows a 512 KB stack and the engine dies with SIGBUS
// (observed in real lichess games as exit code -10 -> the bot restarts the
// engine while the clock runs -> Eclipse flags). A 16 MB stack matches a
// generous main-thread stack and leaves ample headroom above the kMaxSearchPly
// recursion ceiling that now also bounds the depth directly.
class BigThread {
public:
    static constexpr std::size_t kStackBytes = 16 * 1024 * 1024;

    BigThread() = default;

    template <class F, class... Args>
    explicit BigThread(F&& f, Args&&... args) {
        auto* call = new std::function<void()>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, kStackBytes);
        if (pthread_create(&th_, &attr, &trampoline, call) == 0) {
            joinable_ = true;
        } else {
            // Spawn failed: run synchronously rather than silently drop the work.
            (*call)();
            delete call;
        }
        pthread_attr_destroy(&attr);
    }

    BigThread(BigThread&& o) noexcept : th_(o.th_), joinable_(o.joinable_) {
        o.joinable_ = false;
    }
    BigThread& operator=(BigThread&& o) noexcept {
        if (this != &o) {
            if (joinable_) pthread_join(th_, nullptr);
            th_        = o.th_;
            joinable_  = o.joinable_;
            o.joinable_ = false;
        }
        return *this;
    }
    BigThread(const BigThread&)            = delete;
    BigThread& operator=(const BigThread&) = delete;

    ~BigThread() { if (joinable_) pthread_join(th_, nullptr); }

    bool joinable() const noexcept { return joinable_; }
    void join() { if (joinable_) { pthread_join(th_, nullptr); joinable_ = false; } }

private:
    static void* trampoline(void* p) {
        auto* call = static_cast<std::function<void()>*>(p);
        (*call)();
        delete call;
        return nullptr;
    }

    pthread_t th_{};
    bool      joinable_ = false;
};

}  // namespace eclipse
