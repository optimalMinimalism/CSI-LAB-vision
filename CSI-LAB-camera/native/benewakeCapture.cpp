// Linux UART telemetry for Benewake standard 9-byte/cm output. No sensor commands.
// Protocol reference: Benewake TF02-Pro, TF-Luna and TFmini-S user manuals.
#include <array>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <termios.h>
#include <time.h>
#include <unistd.h>

namespace {
volatile std::sig_atomic_t stopping = 0;
void stop(int) { stopping = 1; }
std::int64_t ns(clockid_t clock) {
    timespec t{};
    if (clock_gettime(clock, &t) != 0) throw std::runtime_error("clock_gettime failed");
    return static_cast<std::int64_t>(t.tv_sec) * 1000000000LL + t.tv_nsec;
}
struct FD {
    int fd;
    explicit FD(int value) : fd(value) {}
    ~FD() { if (fd >= 0) ::close(fd); }
    FD(const FD&) = delete;
    FD& operator=(const FD&) = delete;
};
struct OutputMode {
    int original = fcntl(STDOUT_FILENO, F_GETFL);
    OutputMode() {
        if (original < 0 || fcntl(STDOUT_FILENO, F_SETFL, original | O_NONBLOCK) < 0)
            throw std::runtime_error("cannot configure output");
    }
    ~OutputMode() { fcntl(STDOUT_FILENO, F_SETFL, original); }
};
void emit(const std::string& line) {
    const auto deadline = ns(CLOCK_MONOTONIC) + 2000000000LL;
    std::size_t offset = 0;
    while (offset < line.size() && !stopping) {
        pollfd output{STDOUT_FILENO, POLLOUT, 0};
        const int ready = poll(&output, 1, 100);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0 || (output.revents & (POLLERR | POLLHUP | POLLNVAL)))
            throw std::runtime_error("telemetry output closed");
        if (ns(CLOCK_MONOTONIC) > deadline) throw std::runtime_error("telemetry output timeout");
        if (!(output.revents & POLLOUT)) continue;
        const auto written = write(STDOUT_FILENO, line.data() + offset, line.size() - offset);
        if (written < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (written <= 0) throw std::runtime_error("telemetry write failed");
        offset += static_cast<std::size_t>(written);
    }
}
void state(const char* value) { emit(std::string("{\"state\":\"") + value + "\"}\n"); }
void delay() {
    for (int i = 0; i < 10 && !stopping; ++i) poll(nullptr, 0, 100);
}
speed_t speed(int baud) {
    switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
    default: throw std::runtime_error("unsupported baud rate");
    }
}
struct Decoder {
    std::array<unsigned char, 9> bytes{};
    std::size_t size = 0;
    std::uint64_t sequence = 0, checksumErrors = 0, invalid = 0;
    double minimum, maximum;
    unsigned int minimumStrength = 100;
    void consume(unsigned char byte, std::int64_t receivedMono, std::int64_t receivedWall) {
        bytes[size++] = byte;
        // Preserve a partial header and recover one byte at a time after corruption.
        while (size && (bytes[0] != 0x59 || (size >= 2 && bytes[1] != 0x59))) shift();
        if (size != bytes.size()) return;
        unsigned int sum = 0;
        for (std::size_t i = 0; i < 8; ++i) sum += bytes[i];
        if ((sum & 255) != bytes[8]) {
            ++checksumErrors;
            shift();
            emit("{\"state\":\"invalid\",\"invalid_measurements\":" + std::to_string(invalid)
                 + ",\"checksum_errors\":" + std::to_string(checksumErrors) + "}\n");
            return;
        }
        const double distance = (bytes[2] | (bytes[3] << 8)) / 100.0;
        const unsigned int strength = bytes[4] | (bytes[5] << 8);
        const double temperature = (bytes[6] | (bytes[7] << 8)) / 8.0 - 256.0;
        size = 0;
        if (distance < minimum || distance > maximum || strength < minimumStrength || strength == 65535) {
            ++invalid;
            emit("{\"state\":\"invalid\",\"invalid_measurements\":" + std::to_string(invalid)
                 + ",\"checksum_errors\":" + std::to_string(checksumErrors) + "}\n");
            return;
        }
        ++sequence;
        emit("{\"state\":\"reading\",\"sequence\":" + std::to_string(sequence)
             + ",\"distance_m\":" + std::to_string(distance)
             + ",\"signal_strength\":" + std::to_string(strength)
             + ",\"temperature_c\":" + std::to_string(temperature)
             + ",\"received_monotonic_ns\":" + std::to_string(receivedMono)
             + ",\"received_timestamp_ns\":" + std::to_string(receivedWall)
             + ",\"checksum_errors\":" + std::to_string(checksumErrors)
             + ",\"invalid_measurements\":" + std::to_string(invalid) + "}\n");
    }
    void shift() {
        for (std::size_t i = 1; i < size; ++i) bytes[i - 1] = bytes[i];
        --size;
    }
};
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: benewakeCapture DEVICE BAUD tf02-pro|tf-luna|tfmini-s\n";
            return 2;
        }
        char* end = nullptr;
        const long baud = std::strtol(argv[2], &end, 10);
        if (!*argv[2] || *end || baud < 9600 || baud > 230400)
            throw std::runtime_error("invalid baud rate");
        const auto serialSpeed = speed(static_cast<int>(baud));
        const std::string model = argv[3];
        if (model != "tf02-pro" && model != "tf-luna" && model != "tfmini-s")
            throw std::runtime_error("unsupported model");
        OutputMode outputMode;
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        std::signal(SIGPIPE, SIG_IGN);
        Decoder decoder{};
        decoder.minimum = model == "tf-luna" ? 0.2 : 0.1;
        decoder.maximum = model == "tf02-pro" ? 40.0 : model == "tf-luna" ? 8.0 : 12.0;
        // TF02-Pro: conservatively reject strength <=60, plus saturated returns.
        // Its 4500/65534/65535 cm error values also exceed the 40 m acceptance limit.
        decoder.minimumStrength = model == "tf02-pro" ? 61 : 100;
        while (!stopping) {
            FD device(open(argv[1], O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC));
            if (device.fd < 0) { state("disconnected"); delay(); continue; }
            termios options{};
            if (tcgetattr(device.fd, &options) < 0) { state("disconnected"); delay(); continue; }
            cfmakeraw(&options);
            options.c_cflag &= ~(PARENB | CSTOPB | CSIZE | CRTSCTS);
            options.c_cflag |= CS8 | CLOCAL | CREAD;
            options.c_cc[VMIN] = 0;
            options.c_cc[VTIME] = 0;
            if (cfsetispeed(&options, serialSpeed) < 0 || cfsetospeed(&options, serialSpeed) < 0 ||
                tcsetattr(device.fd, TCSANOW, &options) < 0 || tcflush(device.fd, TCIFLUSH) < 0) {
                state("disconnected"); delay(); continue;
            }
            decoder.size = 0;
            state("connected");
            while (!stopping) {
                pollfd input{device.fd, POLLIN, 0};
                const int ready = poll(&input, 1, 100);
                if (ready < 0 && errno == EINTR) continue;
                if (ready < 0 || (input.revents & (POLLERR | POLLHUP | POLLNVAL))) break;
                if (!(input.revents & POLLIN)) continue;
                std::array<unsigned char, 256> chunk{};
                const auto count = read(device.fd, chunk.data(), chunk.size());
                const auto mono = ns(CLOCK_MONOTONIC), wall = ns(CLOCK_REALTIME);
                if (count < 0 && (errno == EINTR || errno == EAGAIN)) continue;
                if (count <= 0) break;
                // All packets in one read share a reception time, not an exposure time.
                for (ssize_t i = 0; i < count && !stopping; ++i) decoder.consume(chunk[i], mono, wall);
            }
            if (!stopping) { state("disconnected"); delay(); }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
