#ifndef TIC_TOC_H_
#define TIC_TOC_H_

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
namespace Common {
    class TicToc {
public:
    TicToc() { tic(); }

    void tic() { start_ = std::chrono::system_clock::now(); }

    double toc() {
        end_ = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_ - start_;
        return elapsed_seconds.count() * 1000;
}

private:
std::chrono::time_point<std::chrono::system_clock> start_, end_;
};
}


#endif