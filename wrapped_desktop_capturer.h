#ifndef EXAMPLES_LOCALVIDEO_CAPTURE_LOCALVIDEO_CAPTURER_TEST_H_
#define EXAMPLES_LOCALVIDEO_CAPTURE_LOCALVIDEO_CAPTURER_TEST_H_

#include "api/scoped_refptr.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "examples/peerconnection/localvideo/test_desktop_capturer.h"
#include "api/video/i420_buffer.h"

#include <thread>
#include <atomic>
#include <string>
#include "rtc_tools/video_file_reader.h"
namespace webrtc {

// Helper function to force sender mode
void ForceWrapperSender();

// Threshold for considering frames as static (percentage similarity)
constexpr float kStaticFrameThreshold = 0.95f;  // Reduced from 0.90f for even more aggressive skipping
// Number of static frames before we start skipping
constexpr int kStaticFrameCountThreshold = 3;   // Reduced from 3 to start skipping sooner
// Maximum skip count to ensure periodic updates even in static scenes
constexpr int kMaxFrameSkipCount = 10;          // Increased to allow more skips

class WrappedDesktopCapturer : public TestDesktopCapturer,
                       public rtc::VideoSinkInterface<VideoFrame> {
 public:

  static WrappedDesktopCapturer* Create();

  ~WrappedDesktopCapturer() override;

  void StartCapture();
  void StopCapture();

  void OnFrame(const VideoFrame& frame) override {}

  ::std::string GetWindowTitle() const { return window_title_; }

 private:
  WrappedDesktopCapturer();
  bool Init();
  void Destory();
  
  // Smart frame skipping detection
  float CalculateFrameSimilarity(const I420BufferInterface* current, const I420BufferInterface* previous);
  bool ShouldSkipFrame(const I420BufferInterface* frame_buffer);
  
  // Motion prediction for adaptive frame skipping
  bool PredictMotion();
  
  // Performance monitoring and logging
  void LogPerformanceStats(bool force);

  size_t fps_;
  ::std::string window_title_;

  ::std::unique_ptr<::std::thread> capture_thread_;
  ::std::atomic<bool> start_flag_;

  rtc::scoped_refptr<I420Buffer> i420_buffer_;
  rtc::scoped_refptr<I420Buffer> previous_frame_buffer_;

  rtc::VideoBroadcaster broadcaster_;
  cricket::VideoAdapter video_adapter_;

  // Frame skip statistics
  int static_frame_count_ = 0;
  int frame_skip_count_ = 0;
  int total_frames_skipped_ = 0;
  int consecutive_skips_ = 0;  // Track consecutive skipped frames
  
  int frame_count_ = 0;
  
  // Adaptive frame skipping
  float adaptive_threshold_ = kStaticFrameThreshold;
  float recent_similarities_[5] = {0.0f}; // Store last 5 similarity scores (increased from 3)
  int recent_similarities_index_ = 0;
  
  // Bandwidth and performance tracking
  int64_t avg_frame_size_ = 0;  // Estimated average frame size in bytes
  int64_t total_bandwidth_saved_ = 0;  // Total estimated bandwidth saved
  int64_t last_log_time_ms_ = 0;  // Last time we logged performance stats
  
  // Energy saving mode
  bool energy_saving_mode_ = false;  // Energy saving mode status
};
}  // namespace webrtc

#endif  // EXAMPLES_LOCALVIDEO_CAPTURE_LOCALVIDEO_CAPTURER_TEST_H_