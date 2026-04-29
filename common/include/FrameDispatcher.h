#pragma once
#include <librealsense2/rs.hpp>
#include <mutex>
#include <atomic>


template<typename Frame>
class queque_mutex {
    public:
    void push(Frame&& fs) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_latestFs = std::move(fs);;
        m_sequence++;   
    }
    bool pop(Frame& outFs,int &last_seq) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_sequence > 0) {
            outFs = m_latestFs;
            last_seq = m_sequence;
            return true;
        }
        last_seq = 0;
        return false;
    }


    private:
        std::mutex m_mutex;
        Frame m_latestFs;
        uint64_t m_sequence{0};
};