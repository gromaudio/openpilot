#include "camera_file.h"

#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>
#include <assert.h>

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "buffering.h"

extern volatile sig_atomic_t do_exit;

#define FRAME_WIDTH  853
#define FRAME_HEIGHT 480

unsigned char frame[(FRAME_WIDTH*FRAME_HEIGHT*3)/2];

namespace {
void camera_open(CameraState *s, VisionBuf *camera_bufs, bool rear) {
  assert(camera_bufs);
  s->camera_bufs = camera_bufs;
}

void camera_close(CameraState *s) {
  tbuffer_stop(&s->camera_tb);
}

void camera_release_buffer(void *cookie, int buf_idx) {
  CameraState *s = static_cast<CameraState *>(cookie);
}

void camera_init(CameraState *s, int camera_id, unsigned int fps) {
  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->frame_size = s->ci.frame_height * s->ci.frame_stride;
  s->fps = fps;

  tbuffer_init2(&s->camera_tb, FRAME_BUF_COUNT, "frame", camera_release_buffer, s);
}

static void* rear_thread(void *arg) {
  int err;

  set_thread_name("webcam_rear_thread");
  CameraState* s = (CameraState*)arg;

  uint32_t frame_id = 0;
  TBuffer* tb = &s->camera_tb;



  int x, y, count;
 
  // Create a pointer for each component's chunk within the frame
  // Note that the size of the Y chunk is W*H, but the size of both
  // the U and V chunks is (W/2)*(H/2). i.e. the resolution is halved
  // in the vertical and horizontal directions for U and V.
  unsigned char *lum, *u, *v;
  lum = frame;
  u = frame + FRAME_HEIGHT*FRAME_WIDTH;
  v = u + (FRAME_HEIGHT*FRAME_WIDTH/4);


  // Open an input pipe from ffmpeg and an output pipe to a second instance of ffmpeg
  FILE *pipein = popen("ffmpeg -i /tmp/fcamera.mp4 -vf scale=853:480 -f image2pipe -vcodec rawvideo -pix_fmt yuv420p -", "r");

  // Process video frames
  while(!do_exit)
  {
    // Read a frame from the input pipe into the buffer
    // Note that the full frame size (in bytes) for yuv420p
    // is (W*H*3)/2. i.e. 1.5 bytes per pixel. This is due
    // to the U and V components being stored at lower resolution.
    count = fread(frame, 1, (FRAME_HEIGHT*FRAME_WIDTH*3)/2, pipein);
    fprintf(stderr, "Size: %d, %d\n", count, frame_id);   
    int transformed_size = count;

    const int buf_idx = tbuffer_select(tb);
    s->camera_bufs_metadata[buf_idx] = {
      .frame_id = frame_id,
    };

    cl_command_queue q = s->camera_bufs[buf_idx].copy_q;
    cl_mem yuv_cl = s->camera_bufs[buf_idx].buf_cl;
    cl_event map_event;
    void *yuv_buf = (void *)clEnqueueMapBuffer(q, yuv_cl, CL_TRUE,
                                                CL_MAP_WRITE, 0, transformed_size,
                                                0, NULL, &map_event, &err);
    assert(err == 0);
    clWaitForEvents(1, &map_event);
    clReleaseEvent(map_event);
    memcpy(yuv_buf, frame, transformed_size);

    clEnqueueUnmapMemObject(q, yuv_cl, yuv_buf, 0, NULL, &map_event);
    clWaitForEvents(1, &map_event);
    clReleaseEvent(map_event);
    tbuffer_dispatch(tb, buf_idx);
    
    sleep(1);
    frame_id += 1;
  }

  // Flush and close input and output pipes
  fflush(pipein);
  pclose(pipein);

  return NULL;
}

}  // namespace

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  // road facing
  [CAMERA_ID_LGC920] = {
      .frame_width = FRAME_WIDTH,
      .frame_height = FRAME_HEIGHT,
      .frame_stride = FRAME_WIDTH*3,
      .bayer = false,
      .bayer_flip = false,
  },
};

void cameras_init(DualCameraState *s) {
  memset(s, 0, sizeof(*s));

  camera_init(&s->rear, CAMERA_ID_LGC920, 20);
  camera_init(&s->front, CAMERA_ID_LGC920, 10);
  s->rear.transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  }};
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(DualCameraState *s, VisionBuf *camera_bufs_rear,
                  VisionBuf *camera_bufs_focus, VisionBuf *camera_bufs_stats,
                  VisionBuf *camera_bufs_front) {
  assert(camera_bufs_rear);
  assert(camera_bufs_front);
  int err;


  // LOG("*** open rear ***");
  camera_open(&s->rear, camera_bufs_rear, true);
}

void cameras_close(DualCameraState *s) {
  camera_close(&s->rear);
}

void cameras_run(DualCameraState *s) {
  set_thread_name("webcam_thread");

  int err;
  pthread_t rear_thread_handle;
  err = pthread_create(&rear_thread_handle, NULL,
                        rear_thread, &s->rear);
  assert(err == 0);

  err = pthread_join(rear_thread_handle, NULL);
  assert(err == 0);
  cameras_close(s);
}
