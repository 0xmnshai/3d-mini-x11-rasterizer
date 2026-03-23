
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define INITIAL_WIDTH 800
#define INITIAL_HEIGHT 600

static int g_width = INITIAL_WIDTH;
static int g_height = INITIAL_HEIGHT;

#define MAX_COMMANDS 1024 * 4
#define RING_SIZE 1024 * 8

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

  g_frame_buffer = malloc(w * h * sizeof(uint32_t));
  g_depth_buffer = malloc(w * h * sizeof(float));

  g_ximage = XCreateImage(display, visual, 24, ZPixmap, 0,
                          (char *)g_frame_buffer, w, h, 32, 0);
}

typedef enum { OP_DRAW, OP_CLEAR } OpCode;

typedef struct {
  OpCode op_code;
  Vertex *vertices;
  uint32_t count;
  Mat4 mvp;
} gpu_command;

typedef struct {
  gpu_command commands[1024];
  int count;
} command_buffer;

void gpu_rasterize_triangle(Vertex v0, Vertex v1, Vertex v2) {
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

void gpu_process_commands(gpu_command *ring_buffer, int count) {
  for (int i = 0; i < count; i++) {
    gpu_command cmd = ring_buffer[i];

    if (cmd.op_code == OP_CLEAR) {
      // memset(g_frame_buffer, 0x11, g_width * g_height * sizeof(uint32_t));
      
      uint8_t r = (uint8_t)(g_clear_color.r * 255.0f);
      uint8_t g = (uint8_t)(g_clear_color.g * 255.0f);
      uint8_t b = (uint8_t)(g_clear_color.b * 255.0f);
      uint32_t clear_pixel = (r << 16) | (g << 8) | b;

      for (int i = 0; i < g_height * g_width; i++)
        g_frame_buffer[i] = clear_pixel;
      for (int i = 0; i < g_height * g_width; i++)
        g_depth_buffer[i] = 1.0f;

    } else if (cmd.op_code == OP_DRAW) {
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

        gpu_rasterize_triangle(tri[0], tri[1], tri[2]);
      }
    }
  }
}

void kernel_submit_buffer(command_buffer *cmd) {
  gpu_process_commands(cmd->commands, cmd->count);
  // GPU_doorbell = 1;
  cmd->count = 0;
}

static command_buffer g_command_buffer;
static Mat4 g_current_mvp;

void glClear() {
  g_command_buffer.commands[g_command_buffer.count++].op_code = OP_CLEAR;
}

void glClearColor(float r, float g, float b) {
  g_clear_color.r = r;
  g_clear_color.g = g;
  g_clear_color.b = b;
}

void glDrawArrays(Vertex *verts, uint32_t count) {
  gpu_command *cmd = &g_command_buffer.commands[g_command_buffer.count++];

  cmd->op_code = OP_DRAW;
  cmd->vertices = verts;
  cmd->count = count;
  cmd->mvp = g_current_mvp;
}

void glFlush() { kernel_submit_buffer(&g_command_buffer); }

int main() {

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

    g_current_mvp = mat4_mul(proj, mat4_mul(view, model));

    Vertex cube_buffer[36];
    for (int i = 0; i < 36; i++) {
      cube_buffer[i] = cube[indices[i]];
    }

    glDrawArrays(cube_buffer, 36);
    glFlush();

    XPutImage(display, window, gc, g_ximage, 0, 0, 0, 0, g_width, g_height);
    XFlush(display);

    angle += 0.02f;
    usleep(16000);
  }

  XCloseDisplay(display);

  return 0;
}