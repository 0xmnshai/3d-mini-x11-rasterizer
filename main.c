#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <pthread.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define INITIAL_WIDTH 800
#define INITIAL_HEIGHT 600

static int g_width = INITIAL_WIDTH;
static int g_height = INITIAL_HEIGHT;

#define MAX_COMMANDS 1024 * 4
#define RING_BUFFER_SIZE 1024 * 4
#define GPU_VRAM_SIZE 1024 * 1024 * 64

typedef uint32_t gpu_addr_t;

typedef struct {
  float x, y, z, w;
} Vec4;

typedef struct {
  float x, y, z;
} Vec3;

typedef struct {
  float u, v;
} Vec2;

typedef struct {
  float r, g, b;
} Color;

typedef struct {
  float m[4][4];
} Mat4;

typedef struct {
  Vec3 pos;
  Color color;
} Vertex;

typedef struct {
  uint32_t count;
  uint32_t start;
} pipe_draw_start_count_bias;

typedef struct {
  uint32_t node;
  uint32_t instance_count;
} pipe_draw_info;

Mat4 mat4_identity() {
  Mat4 m = {0};
  for (size_t i = 0; i < 4; i++)
    m.m[i][i] = 1;
  return m;
}

Mat4 mat4_mul(Mat4 a, Mat4 b) {
  Mat4 r = {0};

  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
      for (size_t k = 0; k < 4; k++)
        r.m[i][j] += a.m[i][k] * b.m[k][j];

  return r;
}

Vec4 mat4_mul_vec4(Mat4 m, Vec4 v) {
  return (Vec4){
      m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w,
      m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w,
      m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w,
      m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w,
  };
}

Mat4 mat4_perspective(float fov, float aspect, float near, float far) {
  float fov_rad = fov * (M_PI / 180.0f);
  float t = tanf(fov_rad / 2.0f);

  Mat4 m = {0};
  m.m[0][0] = 1.0f / (aspect * t);
  m.m[1][1] = 1.0f / t;
  m.m[2][2] = -(far + near) / (far - near);
  m.m[2][3] = -(2.0f * far * near) / (far - near);
  m.m[3][2] = -1.0f;
  m.m[3][3] = 0.0f;

  return m;
}

static uint32_t *g_frame_buffer = NULL;
static float *g_depth_buffer = NULL;
static XImage *g_ximage = NULL;
static Color g_clear_color = {0.184f, 0.310f, 0.310f}; // Slate Green

void resize_buffers(Display *display, Visual *visual, int w, int h) {
  if (g_frame_buffer)
    free(g_frame_buffer);
  if (g_depth_buffer)
    free(g_depth_buffer);
  if (g_ximage) {
    g_ximage->data = NULL;
    XDestroyImage(g_ximage);
  }

  g_width = w;
  g_height = h;

  g_frame_buffer = (uint32_t *)malloc(w * h * sizeof(uint32_t));
  g_depth_buffer = (float *)malloc(w * h * sizeof(float));

  // g_ximage = XCreateImage(display, visual, 24, ZPixmap, 0,
  //                         (char *)g_frame_buffer, w, h, 32, 0);

  g_ximage = XCreateImage(display, visual, 24, ZPixmap, 0,
                          (char *)g_frame_buffer, w, h, 32, 0);
}

typedef enum {
  GPU_OP_NOP = 0,
  GPU_OP_CLEAR_COLOR = 1,
  GPU_OP_CLEAR_DEPTH = 2,
  GPU_OP_LOAD_VP_MATRIX = 3,
  GPU_OP_DRAW_PRIMITIVES = 4,
  GPU_OP_BIND_VBO = 5,
  GPI_OP_DRAW = 6
} gpu_opcode_t;

typedef struct {
  gpu_opcode_t op_code;
  Vertex *vertices;
  uint32_t count;
  Mat4 mvp;
} gpu_command;

typedef struct {
  gpu_opcode_t op_code;
  uint32_t arg0;
  uint32_t arg1;
  float f_arg[16];
} gpu_packet_t;

typedef struct {
  uint32_t *framebuffer;
  float *depthbuffer;
  uint8_t *vram;
  gpu_packet_t ring_buffer[RING_BUFFER_SIZE];
  uint32_t read_ptr;
  uint32_t write_ptr;

  Mat4 view_proj;
  gpu_addr_t bound_vbo;
} gpu_hardware_t;

typedef struct {
  gpu_command commands[1024];
  int count;
} command_buffer;

gpu_hardware_t g_gpu;

static float edge_func(Vec4 a, Vec4 b, Vec4 c) {
  return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

void gpu_rasterize(Vertex v0, Vertex v1, Vertex v2) {
  int min_x = fmax(0, floor(fmin(v0.pos.x, fmin(v1.pos.x, v2.pos.x))));
  int min_y = fmax(0, floor(fmin(v0.pos.y, fmin(v1.pos.y, v2.pos.y))));
  int max_x = fmin(g_width, ceil(fmax(v0.pos.x, fmax(v1.pos.x, v2.pos.x))));
  int max_y = fmin(g_height, ceil(fmax(v0.pos.y, fmax(v1.pos.y, v2.pos.y))));

  float area = (v1.pos.x - v0.pos.x) * (v2.pos.y - v0.pos.y) -
               (v2.pos.x - v0.pos.x) * (v1.pos.y - v0.pos.y);

  if (fabs(area) < 0.0001f)
    return;

  for (int y = min_y; y < max_y; y++) {
    for (int x = min_x; x < max_x; x++) {
      float wo =
          ((v1.pos.x - x) * (v2.pos.y - y) - (v1.pos.y - y) * (v2.pos.x - x)) /
          area;
      float w1 =
          ((v2.pos.x - x) * (v0.pos.y - y) - (v2.pos.y - y) * (v0.pos.x - x)) /
          area;
      float w2 = 1.0f - wo - w1;

      if (wo >= 0 && w1 >= 0 && w2 >= 0) {
        float z = wo * v0.pos.z + w1 * v1.pos.z + w2 * v2.pos.z;

        if (z < g_depth_buffer[y * g_width + x]) {
          g_depth_buffer[y * g_width + x] = z;

          uint8_t r =
              (uint8_t)((wo * v0.color.r + w1 * v1.color.r + w2 * v2.color.r) *
                        255.0f);
          uint8_t g =
              (uint8_t)((wo * v0.color.g + w1 * v1.color.g + w2 * v2.color.g) *
                        255.0f);
          uint8_t b =
              (uint8_t)((wo * v0.color.b + w1 * v1.color.b + w2 * v2.color.b) *
                        255.0f);

          g_frame_buffer[y * g_width + x] = (r << 16) | (g << 8) | b;
        }
      }
    }
  }
}

void gpu_hw_rasterize(Vec4 v0, Vec4 v1, Vec4 v2, uint32_t color) {
  // 1. Viewport Transform
  Vec4 p0 = {(v0.x / v0.w + 1) * 0.5f * g_width,
             (1 - (v0.y / v0.w + 1) * 0.5f) * g_height, v0.z / v0.w, 1};
  Vec4 p1 = {(v1.x / v1.w + 1) * 0.5f * g_width,
             (1 - (v1.y / v1.w + 1) * 0.5f) * g_height, v1.z / v1.w, 1};
  Vec4 p2 = {(v2.x / v2.w + 1) * 0.5f * g_width,
             (1 - (v2.y / v2.w + 1) * 0.5f) * g_height, v2.z / v2.w, 1};

  // 2. Bounding Box
  int minX = fmax(0, floor(fmin(p0.x, fmin(p1.x, p2.x))));
  int maxX = fmin(g_width - 1, ceil(fmax(p0.x, fmax(p1.x, p2.x))));
  int minY = fmax(0, floor(fmin(p0.y, fmin(p1.y, p2.y))));
  int maxY = fmin(g_height - 1, ceil(fmax(p0.y, fmax(p1.y, p2.y))));

  float area = edge_func(p0, p1, p2);
  if (area <= 0)
    return; // Backface culling

  for (int y = minY; y <= maxY; y++) {
    for (int x = minX; x <= maxX; x++) {
      Vec4 p = {x + 0.5f, y + 0.5f, 0, 0};
      float w0 = edge_func(p1, p2, p) / area;
      float w1 = edge_func(p2, p0, p) / area;
      float w2 = edge_func(p0, p1, p) / area;

      if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
        float z = w0 * p0.z + w1 * p1.z + w2 * p2.z;
        if (z < g_gpu.depthbuffer[y * g_width + x]) {
          g_gpu.depthbuffer[y * g_width + x] = z;
          g_gpu.framebuffer[y * g_width + x] = color;
        }
      }
    }
  }
}

void *gpu_worker_thread(void *arg) {
  while (1) {
    if (g_gpu.read_ptr != g_gpu.write_ptr) {
      gpu_packet_t pkt = g_gpu.ring_buffer[g_gpu.read_ptr];

      switch (pkt.op_code) {
      case GPU_OP_CLEAR_COLOR:
        for (int i = 0; i < g_width * g_height; i++)
          g_gpu.framebuffer[i] = pkt.arg0;
        break;
      case GPU_OP_CLEAR_DEPTH:
        for (int i = 0; i < g_width * g_height; i++)
          g_gpu.depthbuffer[i] = 1.0f;
        break;
      case GPU_OP_LOAD_VP_MATRIX:
        memcpy(&g_gpu.view_proj, pkt.f_arg, sizeof(Mat4));
        break;
      case GPU_OP_BIND_VBO:
        g_gpu.bound_vbo = pkt.arg0;
        break;
      case GPU_OP_DRAW_PRIMITIVES: {
        float *vdata = (float *)(g_gpu.vram + g_gpu.bound_vbo);
        for (uint32_t i = 0; i < pkt.arg0; i += 3) {
          Vec4 v[3];
          for (int j = 0; j < 3; j++) {
            Vec4 raw = {vdata[(i + j) * 3 + 0], vdata[(i + j) * 3 + 1],
                        vdata[(i + j) * 3 + 2], 1.0f};
            v[j] = mat4_mul_vec4(g_gpu.view_proj, raw);
          }
          gpu_hw_rasterize(v[0], v[1], v[2], pkt.arg1);
        }
      } break;
      default:
        break;
      }
      g_gpu.read_ptr = (g_gpu.read_ptr + 1) % RING_BUFFER_SIZE;
    } else {
      usleep(100); // Wait for doorbell
    }
  }
  return NULL;
}

void gpu_process_commands(gpu_command *ring_buffer, int count) {
  for (int i = 0; i < count; i++) {
    gpu_command cmd = ring_buffer[i];

    if (cmd.op_code == GPU_OP_CLEAR_COLOR) {
      // memset(g_frame_buffer, 0x11, g_width * g_height * sizeof(uint32_t));

      uint8_t r = (uint8_t)(g_clear_color.r * 255.0f);
      uint8_t g = (uint8_t)(g_clear_color.g * 255.0f);
      uint8_t b = (uint8_t)(g_clear_color.b * 255.0f);
      uint32_t clear_pixel = (r << 16) | (g << 8) | b;

      for (int i = 0; i < g_height * g_width; i++)
        g_frame_buffer[i] = clear_pixel;
      for (int i = 0; i < g_height * g_width; i++)
        g_depth_buffer[i] = 1.0f;

    } else if (cmd.op_code == GPI_OP_DRAW) {
      for (uint32_t j = 0; j < cmd.count; j += 3) {
        Vertex tri[3];

        for (int k = 0; k < 3; k++) {
          Vertex v = cmd.vertices[j + k];

          Vec4 clip =
              mat4_mul_vec4(cmd.mvp, (Vec4){v.pos.x, v.pos.y, v.pos.z, 1.0f});

          float w_inv = 1.0f / clip.w;

          tri[k].pos.x = (clip.x * w_inv * 0.5f + 0.5f) * g_width;
          tri[k].pos.y = (1.0f - (clip.y * w_inv * 0.5f + 0.5f)) * g_height;
          tri[k].pos.z = clip.z * w_inv;
          tri[k].color = v.color;
        }

        gpu_rasterize(tri[0], tri[1], tri[2]);
      }
    }
  }
}

void kernel_submit_buffer(command_buffer *cmd) {
  gpu_process_commands(cmd->commands, cmd->count);
  // GPU_doorbell = 1;
  cmd->count = 0;
}

void kernel_submit_ioctl(gpu_packet_t *user_packets, uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    if (user_packets[i].op_code == GPU_OP_BIND_VBO &&
        user_packets[i].arg0 > GPU_VRAM_SIZE) {
      printf("Kernel Panic: Illegal VRAM Access Attempt!\n");
      return;
    }
    g_gpu.ring_buffer[g_gpu.write_ptr] = user_packets[i];
    g_gpu.write_ptr = (g_gpu.write_ptr + 1) % RING_BUFFER_SIZE;
  }
}

typedef struct {
  gpu_packet_t batch[128];
  uint32_t batch_count;
} driver_context_t;

driver_context_t g_driver;

static command_buffer g_command_buffer;
static Mat4 g_current_mvp;

void glFlush() { kernel_submit_buffer(&g_command_buffer); }

void driver_flush() {
  kernel_submit_ioctl(g_driver.batch, g_driver.batch_count);
  g_driver.batch_count = 0;
}

void driver_emit(gpu_opcode_t op, uint32_t a0, uint32_t a1, float *fptr,
                 int fsize) {
  gpu_packet_t *p = &g_driver.batch[g_driver.batch_count++];
  p->op_code = op;
  p->arg0 = a0;
  p->arg1 = a1;
  if (fptr)
    memcpy(p->f_arg, fptr, fsize);
  if (g_driver.batch_count >= 120)
    driver_flush();
}

uint32_t glGenBuffers() {
  static uint32_t vram_ptr = 0;
  uint32_t handle = vram_ptr;
  vram_ptr += 1024 * 64; // Allocate block
  return handle;
}

void glBufferData(uint32_t handle, void *data, size_t size) {
  memcpy(g_gpu.vram + handle, data, size);
}

void glBindBuffer(uint32_t handle) {
  driver_emit(GPU_OP_BIND_VBO, handle, 0, NULL, 0);
}

void glClear() {
  g_command_buffer.commands[g_command_buffer.count++].op_code =
      GPU_OP_CLEAR_COLOR;
}

void glClearColor(float r, float g, float b) {
  g_clear_color.r = r;
  g_clear_color.g = g;
  g_clear_color.b = b;
}

// void glClearColor(uint32_t hex) {
//   driver_emit(GPU_OP_CLEAR_COLOR, hex, 0, NULL, 0);
//   driver_emit(GPU_OP_CLEAR_DEPTH, 0, 0, NULL, 0);
// }

void glDrawArrays(Vertex *verts, uint32_t count) {
  gpu_command *cmd = &g_command_buffer.commands[g_command_buffer.count++];

  cmd->op_code = GPI_OP_DRAW;
  cmd->vertices = verts;
  cmd->count = count;
  cmd->mvp = g_current_mvp;
}

// void glDrawArrays(uint32_t count, uint32_t color) {
//   driver_emit(GPU_OP_DRAW_PRIMITIVES, count, color, NULL, 0);
// }

int main() {

  g_gpu.vram = (uint8_t *)calloc(1, GPU_VRAM_SIZE);
  g_gpu.framebuffer = (uint32_t *)calloc(g_width * g_height, 4);
  g_gpu.depthbuffer = (float *)malloc(g_width * g_height * sizeof(float));

  pthread_t gpu_thread;
  pthread_create(&gpu_thread, NULL, gpu_worker_thread, NULL);

  const char *display_name = NULL;

  Display *display = XOpenDisplay(display_name);

  if (!display) {
    printf("Failed to open display\n");
    return 1;
  }

  int screen = XDefaultScreen(display);

  Window root_window = XRootWindow(display, screen);

  Window window = XCreateSimpleWindow(display, root_window, 0, 0, INITIAL_WIDTH,
                                      INITIAL_HEIGHT, 1, 0, 0);

  XSelectInput(display, window,
               ExposureMask | KeyPressMask | StructureNotifyMask);
  XMapWindow(display, window);

  GC gc = DefaultGC(display, screen);

  Visual *visual = XDefaultVisual(display, screen);

  resize_buffers(display, visual, INITIAL_WIDTH, INITIAL_HEIGHT);
  glClearColor(0.184f, 0.310f, 0.310f); // Slate Green

  Vertex cube[] = {
      {{-0.5, -0.5, 0.5}, {1, 0, 0}},  {{0.5, -0.5, 0.5}, {0, 1, 0}},
      {{0.5, 0.5, 0.5}, {0, 0, 1}},    {{-0.5, 0.5, 0.5}, {1, 1, 0}},
      {{-0.5, -0.5, -0.5}, {1, 0, 1}}, {{0.5, -0.5, -0.5}, {0, 1, 1}},
      {{0.5, 0.5, -0.5}, {1, 1, 1}},   {{-0.5, 0.5, -0.5}, {0, 0, 0}},
  };

  uint32_t indices[] = {
      0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
      1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
  };

  float angle = 0;

  while (1) {
    while (XPending(display)) {
      XEvent event;
      XNextEvent(display, &event);
      if (event.type == KeyPress)
        break;
      if (event.type == ConfigureNotify) {
        XConfigureEvent xce = event.xconfigure;
        if (xce.width != g_width || xce.height != g_height) {
          resize_buffers(display, visual, xce.width, xce.height);
        }
      }
    }

    glClear();

    // angle += 0.05;
    Mat4 model = mat4_identity();

    model.m[0][0] = cosf(angle);
    model.m[0][2] = sinf(angle);
    model.m[2][0] = -sinf(angle);
    model.m[2][2] = cosf(angle);

    Mat4 view = mat4_identity();
    view.m[2][3] = -2.0f; // Translate block

    Mat4 proj =
        mat4_perspective(45.0f, (float)g_width / g_height, 0.1f, 100.0f);

    // Mat4 mvp = mat4_mul(proj, mat4_mul(view, model));

    float s = sinf(angle), c = cosf(angle);
    Mat4 mvp = {{{c, 0, s, 0},
                 {s * s, c, -s * c, 0},
                 {-c * s, s, c * c, -2.5f}, // Perspective-ish translation
                 {0, 0, 1, 0}}};

    g_current_mvp = mat4_mul(proj, mat4_mul(view, model));

    Vertex cube_buffer[36];
    for (int i = 0; i < 36; i++) {
      cube_buffer[i] = cube[indices[i]];
    }

    // driver_emit(GPU_OP_LOAD_VP_MATRIX, 0, 0, (float *)&mvp, sizeof(mat4));
    // glBindBuffer(vbo);
    // glDrawArrays(24, 0x00ffaa);

    glDrawArrays(cube_buffer, 36);
    glFlush();
    // driver_flush();

    XPutImage(display, window, gc, g_ximage, 0, 0, 0, 0, g_width, g_height);
    XFlush(display);

    angle += 0.02f;
    usleep(16000);
  }

  XCloseDisplay(display);

  return 0;
}
