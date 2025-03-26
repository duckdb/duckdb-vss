//
// Created by root on 6/20/24.
//

#ifndef GRAPH_SEARCH_THREAD_POOL_H
#define GRAPH_SEARCH_THREAD_POOL_H

#include <iostream>
#include <vector>
#include <thread>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <exception>
#include <mutex>

class ThreadPool {
public:
    ThreadPool(size_t numThreads);
    ~ThreadPool();

    void enqueue(std::function<void()> task);
    void waitForCompletion();

private:
    void worker();

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    std::atomic<int> activeTasks;
};



#endif //GRAPH_SEARCH_THREAD_POOL_H
