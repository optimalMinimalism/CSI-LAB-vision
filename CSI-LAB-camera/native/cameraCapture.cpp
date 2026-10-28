// Bounded, streaming capture worker. stdout is exclusively the binary CVF1 protocol.
#include <opencv2/opencv.hpp>
#include <openssl/evp.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static std::int64_t wallNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
static std::int64_t monoNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static std::string sha256(const std::vector<unsigned char>& bytes) {
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int size=0;
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(),EVP_sha256(),nullptr)!=1 ||
        EVP_DigestUpdate(context.get(),bytes.data(),bytes.size())!=1 ||
        EVP_DigestFinal_ex(context.get(),digest,&size)!=1) throw std::runtime_error("SHA-256 failed");
    std::ostringstream text;
    for(unsigned int i=0;i<size;++i) text<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<int>(digest[i]);
    return text.str();
}
static void u32(std::uint32_t value) {
    char bytes[4]; for(int i=0;i<4;++i) bytes[i]=static_cast<char>((value>>(24-8*i))&255);
    std::cout.write(bytes,4);
}
static void send(const cv::Mat& image,std::int64_t timestamp,std::int64_t monotonic,bool synthetic) {
    std::vector<unsigned char> payload;
    if(!cv::imencode(".jpg",image,payload,{cv::IMWRITE_JPEG_QUALITY,85})) throw std::runtime_error("JPEG failed");
    if(payload.size()>8*1024*1024) throw std::runtime_error("Frame exceeds receiver limit");
    std::ostringstream metadata;
    metadata<<"{\"timestamp_ns\":"<<timestamp<<",\"monotonic_ns\":"<<monotonic
            <<",\"width\":"<<image.cols<<",\"height\":"<<image.rows
            <<",\"encoding\":\"JPEG\",\"sha256\":\""<<sha256(payload)
            <<"\",\"timestamp_source\":\""<<(synthetic?"synthetic-native":"opencv-cpp-read-completion")<<"\"}";
    const auto text=metadata.str();
    std::cout.write("CVF1",4);u32(static_cast<std::uint32_t>(text.size()));u32(static_cast<std::uint32_t>(payload.size()));
    std::cout.write(text.data(),static_cast<std::streamsize>(text.size()));
    std::cout.write(reinterpret_cast<const char*>(payload.data()),static_cast<std::streamsize>(payload.size()));
    std::cout.flush();
    if(!std::cout) throw std::runtime_error("Output pipe closed");
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--self-test") {
            send(cv::Mat(48,64,CV_8UC3,cv::Scalar(80,100,120)),wallNs(),monoNs(),true);return 0;
        }
        if(argc!=5) {std::cerr<<"usage: cameraCapture DEVICE WIDTH HEIGHT FPS\n";return 2;}
        const std::string device=argv[1];const int width=std::stoi(argv[2]),height=std::stoi(argv[3]);
        const double fps=std::stod(argv[4]);
        if(width<1 || width>4096 || height<1 || height>2160 || !(fps>=1 && fps<=120)) throw std::runtime_error("Invalid settings");
        cv::VideoCapture capture;
        if(device.find_first_not_of("0123456789")==std::string::npos) capture.open(std::stoi(device),cv::CAP_V4L2);
        else capture.open(device);
        if(!capture.isOpened()) throw std::runtime_error("Cannot open capture source");
        capture.set(cv::CAP_PROP_FRAME_WIDTH,width);capture.set(cv::CAP_PROP_FRAME_HEIGHT,height);
        capture.set(cv::CAP_PROP_FPS,fps);capture.set(cv::CAP_PROP_BUFFERSIZE,1);
        while(true) {
            const auto start=std::chrono::steady_clock::now();cv::Mat image;
            if(!capture.read(image)) break;
            const auto timestamp=wallNs(),monotonic=monoNs();
            send(image,timestamp,monotonic,false);
            std::this_thread::sleep_until(start+std::chrono::nanoseconds(static_cast<std::int64_t>(1e9/fps)));
        }
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
