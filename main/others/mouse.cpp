#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <unistd.h>
#include <libevdev/libevdev.h>

#define SPHERE_RADIUS 150

// センサの距離定数
const double MOUSE_SENSOR_X1DISTANCE = 0.017692852;
const double MOUSE_SENSOR_X2DISTANCE = 0.015048343;
const double MOUSE_SENSOR_Y1DISTANCE = 0.016964967;
const double MOUSE_SENSOR_Y2DISTANCE = 0.018324415;

class MouseTracker {
public:
    MouseTracker(const std::vector<std::string>& device_paths, double k_angle)
        : device_paths(device_paths), k_angle(k_angle), keep_running(true), track_movement(false) {}

    void toggle_tracking() {
        track_movement = !track_movement;
        if (track_movement) {
            std::cout << "Tracking started." << std::endl;
            setup_data_file();
        } else {
            std::cout << "Tracking stopped." << std::endl;
            close_data_file();
        }
    }

    void setup_data_file() {
        std::string directory = "/home/swarm/Documents/python_ANTAM";
        std::system(("mkdir -p " + directory).c_str());

        auto timestamp = std::chrono::system_clock::now();
        std::time_t timestamp_time = std::chrono::system_clock::to_time_t(timestamp);
        char timestamp_str[100];
        std::strftime(timestamp_str, sizeof(timestamp_str), "%Y-%m-%d_%H-%M-%S", std::localtime(&timestamp_time));

        std::string filename = directory + "/Motor_30Hz.csv";
        data_file.open(filename);
        data_file << "Timestamp,X,Y,Z\n";
    }

    void close_data_file() {
        if (data_file.is_open()) {
            data_file.close();
        }
    }

    void read_mouse(const std::string& device_path, int mouse_index) {
        struct libevdev *dev = nullptr;
        int fd = open(device_path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "Failed to open device: " << device_path << std::endl;
            return;
        }

        if (libevdev_new_from_fd(fd, &dev) < 0) {
            std::cerr << "Failed to initialize libevdev for device: " << device_path << std::endl;
            close(fd);
            return;
        }

        double x = 0, y = 0;
        double angle = angle_offset[mouse_index - 1];
        double distance_factor = 1;

        // デバイスに応じて距離定数を設定
        if (device_path.find("event12") != std::string::npos) {
            distance_factor = MOUSE_SENSOR_Y1DISTANCE;
        } else if (device_path.find("event16") != std::string::npos) {
            distance_factor = MOUSE_SENSOR_X1DISTANCE;
        } else if (device_path.find("event20") != std::string::npos) {
            distance_factor = MOUSE_SENSOR_Y2DISTANCE;
        } else if (device_path.find("event24") != std::string::npos) {
            distance_factor = MOUSE_SENSOR_X2DISTANCE;
        }

        while (keep_running) {
            if (!track_movement) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            struct input_event ev;
            if (libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &ev) == 0) {
                if (ev.type == EV_REL) {
                    if (ev.code == REL_X) {
                        y += ev.value;
                    } else if (ev.code == REL_Y) {
                        x -= ev.value;
                    }

                    double get_point_x = y * std::sin(angle) * distance_factor;
                    double get_point_y = y * std::cos(angle) * distance_factor;
                    double get_point_z = x / std::cos(k_angle) * distance_factor;

                    mouse_positions[mouse_index] = {get_point_x, get_point_y, get_point_z};
                }
            }
        }

        libevdev_free(dev);
        close(fd);
    }

    void calculate_average() {
        while (keep_running) {
            if (track_movement) {
                double average_x = 0, average_y = 0, average_z = 0;
                for (const auto& pos : mouse_positions) {
                    average_x += pos.second[0];
                    average_y += pos.second[1];
                    average_z += pos.second[2];
                }
                average_x /= 2;
                average_y /= 2;
                average_z /= mouse_positions.size();

                auto timestamp = std::chrono::system_clock::now();
                auto timestamp_time = std::chrono::system_clock::to_time_t(timestamp);
                char timestamp_str[100];
                std::strftime(timestamp_str, sizeof(timestamp_str), "%Y-%m-%d_%H-%M-%S", std::localtime(&timestamp_time));

                std::lock_guard<std::mutex> lock(file_mutex);
                if (data_file.is_open()) {
                    data_file << timestamp_str << "," << average_x << "," << average_y << "," << average_z << "\n";
                }

                std::cout << "Average Position -> X: " << average_x << ", Y: " << average_y << ", Z: " << average_z << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void start() {
        std::vector<std::thread> threads;

        for (size_t i = 0; i < device_paths.size(); ++i) {
            threads.push_back(std::thread(&MouseTracker::read_mouse, this, device_paths[i], i + 1));
        }

        threads.push_back(std::thread(&MouseTracker::calculate_average, this));

        for (auto& t : threads) {
            t.join();
        }
    }

    void stop() {
        keep_running = false;
        track_movement = false;
        close_data_file();
    }

private:
    std::vector<std::string> device_paths;
    std::map<int, std::vector<double>> mouse_positions;
    double k_angle;
    std::atomic<bool> keep_running;
    bool track_movement;
    std::mutex file_mutex;
    std::ofstream data_file;

    std::vector<double> angle_offset = {0.0, M_PI / 2, M_PI, 3 * M_PI / 2};
};

int main() {
    std::vector<std::string> device_paths = {"/dev/input/event12", "/dev/input/event16", "/dev/input/event20", "/dev/input/event24"};
    MouseTracker tracker(device_paths, M_PI / 6);

    tracker.start();
    return 0;
}
