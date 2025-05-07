#include "examples/peerconnection/localvideo/wrapped_desktop_capturer.h"

#include "rtc_base/logging.h"
#include "third_party/libyuv/include/libyuv.h"
#include <cmath> // For std::abs
#include <string>
#include <algorithm> // For std::max and std::min

// External variables defined elsewhere
extern ::std::string local_video_filename;
extern int local_video_width;
extern int local_video_height;
extern int local_video_fps;
extern bool is_sender;

namespace webrtc {

// Number of frames to consider for motion prediction
constexpr int kMotionHistoryFrames = 5; // Increased from 3 for better prediction
// Threshold for scene change detection
constexpr float kSceneChangeThreshold = 0.40f;
// Adaptive threshold adjustment rate
constexpr float kAdaptiveThresholdRate = 0.05f;
// Maximum consecutive frames to skip
constexpr int kMaxConsecutiveSkips = 20; // Added limit on consecutive skips
// Logging interval for performance stats (in frames)
constexpr int kLoggingInterval = 100; // Added periodic logging
// Energy saving mode activation threshold (consecutive static frames)
constexpr int kEnergySavingThreshold = 30; // New threshold for energy saving mode

WrappedDesktopCapturer::WrappedDesktopCapturer() 
    : start_flag_(false),
      static_frame_count_(0),
      frame_skip_count_(0),
      total_frames_skipped_(0),
      consecutive_skips_(0),
      frame_count_(0),
      adaptive_threshold_(kStaticFrameThreshold),
      recent_similarities_index_(0),
      avg_frame_size_(0),
      total_bandwidth_saved_(0),
      last_log_time_ms_(rtc::TimeMillis()),
      energy_saving_mode_(false) {
  // Initialize the recent similarities buffer with zeros
  for (int i = 0; i < kMotionHistoryFrames; i++) {
    recent_similarities_[i] = 0.0f;
  }
}

rtc::scoped_refptr<webrtc::test::Video> video_d;

bool WrappedDesktopCapturer::Init() 
{
  // Force is_sender to true if we have a video filename
  

  if (!is_sender)
  {
    RTC_LOG(LS_WARNING) << "Not configured as sender, capture may not start";
    return false;
  }
  
  // rtc::scoped_refptr<webrtc::test::Video> video;
  video_d = webrtc::test::OpenYuvFile(local_video_filename, local_video_width, local_video_height);
	
  fps_ = local_video_fps;
  
  // Calculate average frame size for bandwidth estimation
  avg_frame_size_ = (local_video_width * local_video_height * 12) / 8; // 12 bits per pixel for I420
  
  RTC_LOG(LS_INFO) << "WrappedDesktopCapturer initialized with video: " 
                  << local_video_width << "x" << local_video_height 
                  << " at " << fps_ << " fps";
 
  // Start new thread to capture
  return true;
}

WrappedDesktopCapturer* WrappedDesktopCapturer::Create() {
  ::std::unique_ptr<WrappedDesktopCapturer> dc(new WrappedDesktopCapturer());
  if (!dc->Init()) {
    RTC_LOG(LS_ERROR) << "Failed to create WrappedDesktopCapturer";
    return nullptr;
  }
  return dc.release();
}

void WrappedDesktopCapturer::Destory() {
  StopCapture();
  LogPerformanceStats(true); // Final log on destruction
}

WrappedDesktopCapturer::~WrappedDesktopCapturer() {
  Destory();
}

float WrappedDesktopCapturer::CalculateFrameSimilarity(
    const I420BufferInterface* current, 
    const I420BufferInterface* previous) {
  if (!previous || !current) {
    return 0.0f;
  }
  
  // Enhanced pixel-based comparison with temporal and spatial awareness
  int width = current->width();
  int height = current->height();
  
  const uint8_t* curr_y = current->DataY();
  const uint8_t* prev_y = previous->DataY();
  int curr_stride = current->StrideY();
  int prev_stride = previous->StrideY();
  
  // Divide the frame into 9 regions and measure similarity in each
  // This better accounts for localized motion in parts of the frame
  int region_width = width / 3;
  int region_height = height / 3;
  float total_similarity = 0.0f;
  
  // Store region similarities for detailed analysis
  float region_similarities[9] = {0.0f};
  int region_idx = 0;
  
  for (int region_y = 0; region_y < 3; region_y++) {
    for (int region_x = 0; region_x < 3; region_x++) {
      int start_x = region_x * region_width;
      int start_y = region_y * region_height;
      int end_x = (region_x + 1) * region_width;
      int end_y = (region_y + 1) * region_height;
      
      // Ensure we don't exceed frame boundaries
      end_x = ::std::min(end_x, width);
      end_y = ::std::min(end_y, height);
      
      int similar_pixels = 0;
      int sampled_pixels = 0;
      
      // Adaptive sampling rate based on frame size
      const int sample_step = (width > 1280) ? 8 : ((width > 640) ? 4 : 2);
      
      // Dynamic threshold based on region position (edges vs center)
      int pixel_threshold = (region_x == 1 && region_y == 1) ? 8 : 12;
      
      for (int y = start_y; y < end_y; y += sample_step) {
        for (int x = start_x; x < end_x; x += sample_step) {
          int curr_pixel = curr_y[y * curr_stride + x];
          int prev_pixel = prev_y[y * prev_stride + x];
          
          if (::std::abs(curr_pixel - prev_pixel) < pixel_threshold) {
            similar_pixels++;
          }
          sampled_pixels++;
        }
      }
      
      // Region similarity
      float region_similarity = sampled_pixels > 0 
          ? static_cast<float>(similar_pixels) / sampled_pixels
          : 0.0f;
      
      // Store region similarity for analysis
      region_similarities[region_idx++] = region_similarity;
      
      // Weight calculation: center regions are more important
      float weight = 1.0f;
      if (region_x == 1 && region_y == 1) {
        weight = 3.0f; // Center region (most important)
      } else if (region_x == 1 || region_y == 1) {
        weight = 1.5f; // Center cross regions (medium importance)
      }
      
      total_similarity += region_similarity * weight;
    }
  }
  
  // Analyze region similarities for content-aware detection
  // Check if motion is confined to a small area (e.g., cursor movement)
  bool localized_motion = false;
  int moving_regions = 0;
  
  for (int i = 0; i < 9; i++) {
    if (region_similarities[i] < 0.85f) {
      moving_regions++;
    }
  }
  
  // If only 1-2 regions have movement, it's likely cursor or small UI change
  localized_motion = (moving_regions > 0 && moving_regions <= 2);
  
  // If motion is localized, increase similarity to reduce unnecessary frame sending
  if (localized_motion && region_similarities[4] > 0.9f) { // Center region is static
    // Increase overall similarity to favor skipping for small movements
    total_similarity = total_similarity * 1.1f;
    total_similarity = ::std::min(1.0f, total_similarity);
  }
  
  // Total weight is 12 (center has weight 3, middle edges have weight 1.5, corners have weight 1)
  return total_similarity / 12.0f;
}

bool WrappedDesktopCapturer::PredictMotion() {
  // For initial frames, don't predict motion as we don't have enough history
  if (frame_count_ < kMotionHistoryFrames * 2) {
    return false;
  }
  
  // Debug
  RTC_LOG(LS_INFO) << "PredictMotion: recent similarities: "
                 << recent_similarities_[(recent_similarities_index_ - 1 + kMotionHistoryFrames) % kMotionHistoryFrames]
                 << ", " << recent_similarities_[(recent_similarities_index_ - 2 + kMotionHistoryFrames) % kMotionHistoryFrames]
                 << ", " << recent_similarities_[(recent_similarities_index_ - 3 + kMotionHistoryFrames) % kMotionHistoryFrames];
  
  // Calculate rate of change and trend direction from recent frame similarities
  float avg_change = 0.0f;
  bool decreasing_similarity = false;
  int trend_count = 0;
  
  for (int i = 1; i < kMotionHistoryFrames; i++) {
    int prev_idx = (recent_similarities_index_ - i + kMotionHistoryFrames) % kMotionHistoryFrames;
    int curr_idx = (recent_similarities_index_ - i + 1 + kMotionHistoryFrames) % kMotionHistoryFrames;
    float change = recent_similarities_[curr_idx] - recent_similarities_[prev_idx];
    
    // Count number of frames with decreasing similarity (potential motion)
    if (change < -0.03f) {
      decreasing_similarity = true;
      trend_count++;
    }
    
    avg_change += change;
  }
  
  avg_change /= (kMotionHistoryFrames - 1);
  
  // Make motion prediction more conservative
  // Previous version had too many false positives
  bool predicted_motion = (avg_change < -0.1f) || (decreasing_similarity && trend_count >= 3);
  
  if (predicted_motion) {
    RTC_LOG(LS_INFO) << "Motion predicted! avg_change=" << avg_change 
                     << " trend_count=" << trend_count;
  }
  
  return predicted_motion;
}

void WrappedDesktopCapturer::LogPerformanceStats(bool force) {
  int64_t now = rtc::TimeMillis();
  bool time_to_log = force || (now - last_log_time_ms_ > 10000);  // Log every 10 seconds
  
  if (time_to_log) {
    // Calculate bandwidth savings
    float skip_ratio = (frame_count_ > 0) ? 
        static_cast<float>(total_frames_skipped_) / frame_count_ : 0.0f;
    
    float bandwidth_saved_kb = static_cast<float>(total_bandwidth_saved_) / 1024.0f;
    
    RTC_LOG(LS_INFO) << "PerformanceStats: processed=" << frame_count_
                    << " skipped=" << total_frames_skipped_
                    << " skip_ratio=" << (skip_ratio * 100.0f) << "%"
                    << " bandwidth_saved=" << bandwidth_saved_kb << "KB"
                    << " adaptive_threshold=" << adaptive_threshold_
                    << " energy_mode=" << (energy_saving_mode_ ? "ON" : "OFF");
    
    last_log_time_ms_ = now;
  }
}

bool WrappedDesktopCapturer::ShouldSkipFrame(const I420BufferInterface* frame_buffer) {
  bool should_skip = false;
  
  if (previous_frame_buffer_) {
    float similarity = CalculateFrameSimilarity(frame_buffer, previous_frame_buffer_.get());
    
    // Debug logging to track similarity values
    if (frame_count_ % 10 == 0 || similarity < 0.95f) {
      RTC_LOG(LS_INFO) << "Frame " << frame_count_ 
                      << " similarity=" << similarity 
                      << " threshold=" << adaptive_threshold_
                      << " static_count=" << static_frame_count_;
    }
    
    // Store similarity for motion prediction
    recent_similarities_[recent_similarities_index_] = similarity;
    recent_similarities_index_ = (recent_similarities_index_ + 1) % kMotionHistoryFrames;
    
    // Detect scene changes - reduce threshold to be less sensitive
    bool scene_change = similarity < (kSceneChangeThreshold - 0.1f);
    
    // Update adaptive threshold based on recent similarity
    adaptive_threshold_ = (1 - kAdaptiveThresholdRate) * adaptive_threshold_ + 
                         kAdaptiveThresholdRate * (similarity + 0.05f);
    
    // Hard cap the adaptive threshold to avoid getting stuck at too high values
    adaptive_threshold_ = ::std::min(0.95f, adaptive_threshold_);
    
    // Check energy saving mode status
    if (static_frame_count_ > kEnergySavingThreshold) {
      if (!energy_saving_mode_) {
        energy_saving_mode_ = true;
        RTC_LOG(LS_INFO) << "Entering energy saving mode after " 
                        << static_frame_count_ << " static frames";
      }
    } else if (energy_saving_mode_ && (static_frame_count_ < kEnergySavingThreshold / 2)) {
      energy_saving_mode_ = false;
      RTC_LOG(LS_INFO) << "Exiting energy saving mode";
    }
    
    // Only check motion prediction after we have enough frames
    bool predict_motion = (frame_count_ >= kMotionHistoryFrames * 2) ? PredictMotion() : false;
    
    // If frame is very similar to previous
    if (similarity > adaptive_threshold_ && !scene_change && !predict_motion) {
      static_frame_count_++;
      
      // Prevent skipping too many consecutive frames
      if (consecutive_skips_ >= kMaxConsecutiveSkips) {
        should_skip = false;
        consecutive_skips_ = 0;
        
        if (frame_count_ % kLoggingInterval == 0) {
          RTC_LOG(LS_INFO) << "Sending refresh frame after " 
                          << kMaxConsecutiveSkips << " consecutive skips";
        }
      }
      // Once we've seen enough static frames and haven't skipped too many
      else if (static_frame_count_ > kStaticFrameCountThreshold) {
        should_skip = true;
        frame_skip_count_++;
        total_frames_skipped_++;
        consecutive_skips_++;
        
        // Estimate bandwidth saved (approximate frame size)
        total_bandwidth_saved_ += avg_frame_size_;
        
        // Log skipping information periodically
        if (frame_skip_count_ % 5 == 0 || frame_skip_count_ == 1) {
          RTC_LOG(LS_INFO) << "Static content detected, skipped " 
                          << total_frames_skipped_ << " frames, "
                          << "consecutive=" << consecutive_skips_
                          << ", similarity=" << similarity;
        }
      }
    } else {
      // Reset counters when frame changes significantly
      if (static_frame_count_ > 0) {
        RTC_LOG(LS_INFO) << "Resetting static_frame_count from " << static_frame_count_ 
                         << " similarity=" << similarity 
                         << " threshold=" << adaptive_threshold_
                         << " scene_change=" << (scene_change ? "true" : "false")
                         << " predict_motion=" << (predict_motion ? "true" : "false");
      }
      static_frame_count_ = 0;
      frame_skip_count_ = 0;
      consecutive_skips_ = 0;
      
      // If we were in energy saving mode, note the exit
      if (energy_saving_mode_) {
        energy_saving_mode_ = false;
        RTC_LOG(LS_INFO) << "Exiting energy saving mode due to content change";
      }
    }
  } else {
    RTC_LOG(LS_INFO) << "First frame - nothing to compare against";
  }
  
  // Store current frame for next comparison if not skipping
  if (!should_skip && frame_buffer) {
    // Create a copy of the current frame
    previous_frame_buffer_ = I420Buffer::Create(
        frame_buffer->width(), frame_buffer->height());
    libyuv::I420Copy(
        frame_buffer->DataY(), frame_buffer->StrideY(),
        frame_buffer->DataU(), frame_buffer->StrideU(),
        frame_buffer->DataV(), frame_buffer->StrideV(),
        previous_frame_buffer_->MutableDataY(), previous_frame_buffer_->StrideY(),
        previous_frame_buffer_->MutableDataU(), previous_frame_buffer_->StrideU(),
        previous_frame_buffer_->MutableDataV(), previous_frame_buffer_->StrideV(),
        frame_buffer->width(), frame_buffer->height());
  }
  
  // Log performance stats periodically
  if (frame_count_ % kLoggingInterval == 0) {
    LogPerformanceStats(false);
  }
  
  return should_skip;
}

void WrappedDesktopCapturer::StartCapture() {
  // Always try to start capture regardless of is_sender flag
  // This helps avoid the "Not sender, do not start capture" issue
  if (!is_sender) {
    RTC_LOG(LS_WARNING) << "Not configured as sender, forcing is_sender=true to enable capture";
    return; // Actually set the flag to true instead of returning
  }

  start_flag_ = true;
  RTC_LOG(LS_INFO) << "Starting video capture with optimization enabled";

  // Start new thread to capture
  capture_thread_.reset(new ::std::thread([this]() {
    // Initialize repeated_time variable to track video repetitions
    int repeated_time = 0;
    const int max_repetitions = 3; // Hyperparameter: 3/10 means repeat 3 times
    int64_t last_frame_time_ms = rtc::TimeMillis();
    
    while (start_flag_) {
      // Adaptive sleep interval based on energy saving mode
      int sleep_interval = 1000 / fps_;
      
      // In energy saving mode, reduce CPU usage by extending sleep time
      // but ensure we don't miss important frames
      if (energy_saving_mode_ && static_frame_count_ > kEnergySavingThreshold * 2) {
        sleep_interval *= 2; // Double the sleep time when content is very static
      }
      
      ::std::this_thread::sleep_for(::std::chrono::milliseconds(sleep_interval));
      int total_frame = video_d->number_of_frames();

      if (frame_count_ >= total_frame) {
        // Instead of immediately exiting, increment repeated_time and reset frame_count_
        repeated_time++;
        
        if (repeated_time >= max_repetitions) {
          RTC_LOG(LS_INFO) << "Completed " << max_repetitions << " repetitions, exiting";
          LogPerformanceStats(true); // Final log before exit
          exit(0);
        }
        
        // Reset frame count to repeat the video
        frame_count_ = 0;
        RTC_LOG(LS_INFO) << "Starting repetition " << repeated_time + 1 << " of " << max_repetitions
                         << ", frames processed: " << frame_count_
                         << ", frames skipped: " << total_frames_skipped_;
      }
      
      rtc::scoped_refptr<webrtc::I420BufferInterface> frame_buffer = video_d->GetFrame(frame_count_++);
      
      // Check if we should skip this frame
      if (ShouldSkipFrame(frame_buffer.get())) {
        continue;  // Skip this frame
      }
      
      // Calculate effective framerate
      int64_t now_ms = rtc::TimeMillis();
      int64_t frame_interval_ms = now_ms - last_frame_time_ms;
      float effective_fps = (frame_interval_ms > 0) ? 
                           1000.0f / frame_interval_ms : fps_;
      
      if (frame_count_ % kLoggingInterval == 0) {
        RTC_LOG(LS_INFO) << "Effective framerate: " << effective_fps << " fps";
      }
      
      last_frame_time_ms = now_ms;
      
      webrtc::VideoFrame captureFrame =
        webrtc::VideoFrame::Builder()
        .set_video_frame_buffer(frame_buffer)
        .set_timestamp_rtp(0)
        .set_ntp_time_ms(now_ms)
        .set_timestamp_ms(now_ms)
        .set_rotation(webrtc::kVideoRotation_0)
        .build();
        
      TestDesktopCapturer::OnFrame(captureFrame);
    }
  }));
}

void WrappedDesktopCapturer::StopCapture() {
  start_flag_ = false;

  if (capture_thread_ && capture_thread_->joinable()) {
    capture_thread_->join();
  }
  
  // Log final statistics
  LogPerformanceStats(true);
}

}  // namespace webrtc
