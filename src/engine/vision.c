// Images for the vision encoder, as inference/image_processor.py prepares
// them: decoding, the grid an image gets, Pillow's bicubic resize and gray
// padding (ImageOps.pad), the ViT's patches and 2D RoPE angles, and the
// tokens of the image's span in a prompt.
#include "../model/deepseek/model.h"
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <jpeglib.h>
#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#include "myds4.h"

// inference/config.json: vision_patch_size, vision_downsample_ratio,
// vision_max_n_token, vision_min_pixels (vision_max_wh_ratio is unset).
#define PATCH 14
#define RATIO 3
#define MAX_TOKENS 1024
#define MIN_PIXELS 295936
#define BOMB 178956970  // Pillow refuses images of more pixels (twice Media.MAX_IMAGE_PIXELS)

// ---- decoding: the pixels as Pillow's convert("RGB") gives them ----

// From the image's own pixel data: alpha dropped (not composited), gray
// repeated, palettes looked up, 16-bit channels by their high byte; 0 for
// formats outside these.
static int raw_rgb(CGImageRef img, uint8_t *out, size_t w, size_t h) {
  CGColorSpaceRef cs = CGImageGetColorSpace(img);
  CGColorSpaceModel model = cs ? CGColorSpaceGetModel(cs) : kCGColorSpaceModelUnknown;
  CGBitmapInfo info = CGImageGetBitmapInfo(img);
  CGImageAlphaInfo alpha = (CGImageAlphaInfo)(info & kCGBitmapAlphaInfoMask);
  size_t bpc = CGImageGetBitsPerComponent(img), bpp = CGImageGetBitsPerPixel(img), stride = CGImageGetBytesPerRow(img);
  int indexed = model == kCGColorSpaceModelIndexed;
  int color = model == kCGColorSpaceModelRGB ? 3 : model == kCGColorSpaceModelMonochrome || indexed ? 1 : 0;
  int extra = alpha != kCGImageAlphaNone;  // an alpha or padding component
  if (!color || (info & kCGBitmapFloatComponents) || alpha == kCGImageAlphaOnly || bpp != (color + extra) * bpc) return 0;
  if (indexed ? extra || (bpc != 1 && bpc != 2 && bpc != 4 && bpc != 8) : bpc != 8 && bpc != 16) return 0;
  uint8_t table[256 * 3] = {0};
  if (indexed) {
    CGColorSpaceRef base = CGColorSpaceGetBaseColorSpace(cs);
    if (!base || CGColorSpaceGetModel(base) != kCGColorSpaceModelRGB || CGColorSpaceGetColorTableCount(cs) > 256) return 0;
    CGColorSpaceGetColorTable(cs, table);
  }
  CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(img));
  if (!data) return 0;
  if ((size_t)CFDataGetLength(data) < stride * (h - 1) + (w * bpp + 7) / 8) {
    CFRelease(data);
    return 0;
  }
  const uint8_t *p = CFDataGetBytePtr(data);
  uint32_t order = info & kCGBitmapByteOrderMask;
  int first = alpha == kCGImageAlphaFirst || alpha == kCGImageAlphaPremultipliedFirst || alpha == kCGImageAlphaNoneSkipFirst;
  int premul = alpha == kCGImageAlphaPremultipliedFirst || alpha == kCGImageAlphaPremultipliedLast;
  int comps = color + extra, bytes = (int)bpc / 8;
  // little-endian 16- and 32-bit pixels of 8-bit components hold them in reverse
  int reversed = bpc == 8 && ((bpp == 32 && order == kCGBitmapByteOrder32Little) ||
                              (bpp == 16 && order == kCGBitmapByteOrder16Little));
  int little = bpc == 16 && order == kCGBitmapByteOrder16Little;
  for (size_t y = 0; y < h; y++) {
    const uint8_t *row = p + y * stride;
    for (size_t x = 0; x < w; x++) {
      uint8_t *o = out + (y * w + x) * 3;
      if (indexed) {
        size_t bit = x * bpc;
        memcpy(o, table + 3 * ((row[bit / 8] >> (8 - bpc - bit % 8)) & ((1u << bpc) - 1)), 3);
        continue;
      }
      int v[4];  // the components in their declared order
      for (int c = 0; c < comps; c++) {
        const uint8_t *q = row + (x * comps + (reversed ? comps - 1 - c : c)) * bytes;
        int u = bytes == 1 ? q[0] : little ? q[0] | q[1] << 8 : q[0] << 8 | q[1];
        // 16-bit gray without alpha opens in Pillow as I;16, whose conversion
        // clamps at 255; the other 16-bit images keep their high byte
        v[c] = bytes == 1 ? u : color == 1 && !extra ? (u > 255 ? 255 : u) : u >> 8;
      }
      int a = extra ? v[first ? 0 : color] : 255, *cv = v + (extra && first);
      for (int c = 0; c < 3; c++) {
        int u = cv[color == 3 ? c : 0];
        if (premul) u = a ? (u * 255 + a / 2) / a : 0;
        o[c] = (uint8_t)(u > 255 ? 255 : u);
      }
    }
  }
  CFRelease(data);
  return 1;
}

// Other formats (floating-point, CMYK, packed): drawn into sRGB.
static void draw_rgb(CGImageRef img, uint8_t *out, size_t w, size_t h) {
  CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  uint8_t *buf = calloc(w * h, 4);
  CGContextRef ctx = buf ? CGBitmapContextCreate(buf, w, h, 8, w * 4, cs, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big) : 0;
  if (!ctx) die("cannot draw an image of %zu x %zu pixels", w, h);
  CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
  for (size_t i = 0; i < w * h; i++) memcpy(out + i * 3, buf + i * 4, 3);
  CGContextRelease(ctx);
  CGColorSpaceRelease(cs);
  free(buf);
}

typedef struct { int w, h; uint8_t *p; } Rgb;  // rows of w RGB pixels

// JPEG files through libjpeg-turbo with its defaults (the accurate integer
// IDCT, smooth chroma upsampling), as Pillow decodes them: macOS's decoder
// gave pixels up to 4 levels apart on photos. CMYK and YCCK files are left
// to ImageIO (w = 0).
typedef struct { struct jpeg_error_mgr mgr; jmp_buf back; } JpegErr;
static void jpeg_fail(j_common_ptr c) { longjmp(((JpegErr *)c->err)->back, 1); }

static Rgb decode_jpeg(const void *bytes, size_t len, const char **err) {
  struct jpeg_decompress_struct c;
  JpegErr je;
  static Rgb r;  // static: kept across the longjmp (images are decoded on one thread at a time)
  r = (Rgb){0};
  c.err = jpeg_std_error(&je.mgr);
  je.mgr.error_exit = jpeg_fail;
  if (setjmp(je.back)) {
    jpeg_destroy_decompress(&c);
    free(r.p);
    *err = "the JPEG image could not be decoded";
    return (Rgb){0};
  }
  jpeg_create_decompress(&c);
  jpeg_mem_src(&c, bytes, (unsigned long)len);
  jpeg_read_header(&c, TRUE);
  if (c.jpeg_color_space == JCS_CMYK || c.jpeg_color_space == JCS_YCCK) {
    jpeg_destroy_decompress(&c);
    return (Rgb){0};
  }
  if ((double)c.image_width * c.image_height > BOMB) {
    jpeg_destroy_decompress(&c);
    *err = "the image has more than 178,956,970 pixels";
    return (Rgb){0};
  }
  c.out_color_space = JCS_RGB;
  jpeg_start_decompress(&c);
  r = (Rgb){(int)c.output_width, (int)c.output_height, xmalloc((size_t)c.output_width * c.output_height * 3)};
  while (c.output_scanline < c.output_height) {
    JSAMPROW row = r.p + (size_t)c.output_scanline * c.output_width * 3;
    jpeg_read_scanlines(&c, &row, 1);
  }
  jpeg_finish_decompress(&c);
  jpeg_destroy_decompress(&c);
  return r;
}

static Rgb decode(const void *bytes, size_t len, const char **err) {
  *err = 0;
  if (len > 3 && !memcmp(bytes, "\xff\xd8\xff", 3)) {
    Rgb r = decode_jpeg(bytes, len, err);
    if (r.p || *err) return r;
  }
  CFDataRef d = CFDataCreate(0, bytes, (CFIndex)len);
  CGImageSourceRef src = d ? CGImageSourceCreateWithData(d, 0) : 0;
  CGImageRef img = src && CGImageSourceGetCount(src) ? CGImageSourceCreateImageAtIndex(src, 0, 0) : 0;
  Rgb r = {0};
  size_t w = img ? CGImageGetWidth(img) : 0, h = img ? CGImageGetHeight(img) : 0;
  if (!img) *err = "the image could not be decoded";
  else if (!w || !h) *err = "the image is empty";
  else if ((double)w * h > BOMB) *err = "the image has more than 178,956,970 pixels";
  else {
    r = (Rgb){(int)w, (int)h, xmalloc(w * h * 3)};
    if (!raw_rgb(img, r.p, w, h)) draw_rgb(img, r.p, w, h);
  }
  if (img) CGImageRelease(img);
  if (src) CFRelease(src);
  if (d) CFRelease(d);
  return r;
}

// ---- the grid (plan_image_grid, safe_resize, solve_resize_ratio, llm_grid) ----

static int tokens_of(int th, int tw) { return th * (tw + 1) + 2; }

static void grid(int bh, int bw, int *th, int *tw) {
  *th = (int)ceil((double)(bh / PATCH) / RATIO), *tw = (int)ceil((double)(bw / PATCH) / RATIO);
}

// The largest aspect-preserving size whose grid fits in MAX_TOKENS.
static void fit(int height, int width, int *bh, int *bw) {
  double r = (double)height / width, mw = sqrt((MAX_TOKENS - 2) / r + 0.25) - 0.5, mh = mw * r;
  int cell = PATCH * RATIO;
  if (mw < 1.0) {  // very tall: a single column
    *bh = (MAX_TOKENS - 2) / 2 * cell, *bw = cell;
    return;
  }
  if (mh < 1.0) {  // very wide: a single row
    *bh = cell, *bw = (MAX_TOKENS - 3) * cell;
    return;
  }
  double beta = fmin(floor(mw) * cell / width, floor(mh) * cell / height);
  *bh = (int)floor(height * beta / PATCH) * PATCH, *bw = (int)floor(width * beta / PATCH) * PATCH;
}

// The pixel size an image is padded to (bw x bh) and its token grid; images
// of fewer than MIN_PIXELS are scaled up first.
static void plan(int width, int height, int *bw, int *bh, int *th, int *tw) {
  if ((double)width * height < MIN_PIXELS) {
    double ratio = pow((double)MIN_PIXELS / ((double)width * height), 0.5);
    width = (int)(width * ratio), height = (int)(height * ratio);
  }
  *bw = (int)ceil((double)width / PATCH) * PATCH, *bh = (int)ceil((double)height / PATCH) * PATCH;
  grid(*bh, *bw, th, tw);
  if (tokens_of(*th, *tw) > MAX_TOKENS) {
    fit(height, width, bh, bw);
    grid(*bh, *bw, th, tw);
  }
}

// ---- Pillow's bicubic resize (libImaging/Resample.c, 8 bits per channel) ----

#define FRAC 22  // PRECISION_BITS: the weights' fraction bits

static double cubic(double x) {  // a = -0.5, support 2
  const double a = -0.5;
  x = fabs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

// Each output pixel's taps along one axis (precompute_coeffs and
// normalize_coeffs_8bpc): the first input pixel and their count in bounds,
// their weights in fixed point in k, ksize per pixel.
static int taps(int in, int out, int **bounds, int32_t **k) {
  double scale = (double)in / out, fs = scale < 1 ? 1 : scale, support = 2 * fs;
  int ksize = (int)ceil(support) * 2 + 1;
  double *w = xmalloc((size_t)out * ksize * sizeof *w);
  *bounds = xmalloc((size_t)out * 2 * sizeof **bounds);
  *k = xmalloc((size_t)out * ksize * sizeof **k);
  for (int xx = 0; xx < out; xx++) {
    double center = (xx + 0.5) * scale, ww = 0, ss = 1.0 / fs, *kx = w + (size_t)xx * ksize;
    int xmin = (int)(center - support + 0.5), xmax = (int)(center + support + 0.5);
    if (xmin < 0) xmin = 0;
    if (xmax > in) xmax = in;
    xmax -= xmin;
    for (int x = 0; x < xmax; x++) ww += kx[x] = cubic((x + xmin - center + 0.5) * ss);
    for (int x = 0; x < ksize; x++) kx[x] = x >= xmax ? 0 : ww != 0 ? kx[x] / ww : kx[x];
    (*bounds)[2 * xx] = xmin, (*bounds)[2 * xx + 1] = xmax;
  }
  for (size_t i = 0; i < (size_t)out * ksize; i++)
    (*k)[i] = (int32_t)(w[i] < 0 ? -0.5 + w[i] * (1 << FRAC) : 0.5 + w[i] * (1 << FRAC));
  free(w);
  return ksize;
}

static uint8_t clip8(int32_t v) {
  v >>= FRAC;
  return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

// ImagingResampleInner: along rows first, over the rows the column pass
// reads, then along columns; each pass rounds to 8 bits.
static Rgb resample(Rgb s, int w, int h) {
  int *bx = 0, *by;
  int32_t *kx = 0, *ky;
  int ksx = w != s.w ? taps(s.w, w, &bx, &kx) : 0, ksy = taps(s.h, h, &by, &ky);
  int y0 = by[0], y1 = by[2 * (h - 1)] + by[2 * (h - 1) + 1];
  Rgb t = s;
  if (w != s.w) {
    for (int y = 0; y < h; y++) by[2 * y] -= y0;
    t = (Rgb){w, y1 - y0, xmalloc((size_t)w * (y1 - y0) * 3)};
    for (int y = y0; y < y1; y++)
      for (int x = 0; x < w; x++) {
        const uint8_t *in = s.p + ((size_t)y * s.w + bx[2 * x]) * 3;
        const int32_t *k = kx + (size_t)x * ksx;
        int32_t a[3] = {1 << (FRAC - 1), 1 << (FRAC - 1), 1 << (FRAC - 1)};
        for (int i = 0; i < bx[2 * x + 1]; i++)
          for (int c = 0; c < 3; c++) a[c] += in[3 * i + c] * k[i];
        for (int c = 0; c < 3; c++) t.p[((size_t)(y - y0) * w + x) * 3 + c] = clip8(a[c]);
      }
  }
  Rgb o = t;
  if (h != s.h) {
    o = (Rgb){t.w, h, xmalloc((size_t)t.w * h * 3)};
    for (int y = 0; y < h; y++) {
      const int32_t *k = ky + (size_t)y * ksy;
      for (int x = 0; x < t.w; x++) {
        int32_t a[3] = {1 << (FRAC - 1), 1 << (FRAC - 1), 1 << (FRAC - 1)};
        for (int i = 0; i < by[2 * y + 1]; i++)
          for (int c = 0; c < 3; c++) a[c] += t.p[((size_t)(by[2 * y] + i) * t.w + x) * 3 + c] * k[i];
        for (int c = 0; c < 3; c++) o.p[((size_t)y * t.w + x) * 3 + c] = clip8(a[c]);
      }
    }
    if (t.p != s.p) free(t.p);
  } else if (t.p == s.p) {
    o.p = xmalloc((size_t)s.w * s.h * 3);
    memcpy(o.p, s.p, (size_t)s.w * s.h * 3);
  }
  free(bx), free(kx), free(by), free(ky);
  return o;
}

// Media.resize: a copy at the same size; an image over 100 times taller
// than wide shrinks along its height first.
static Rgb resize(Rgb s, int w, int h) {
  if (s.h > s.w * 100 && h < s.h) {
    Rgb t = resample(s, s.w, h), o = resample(t, w, h);
    free(t.p);
    return o;
  }
  return resample(s, w, h);
}

// ImageOps.pad(image, (bw, bh), color=(127, 127, 127)): contain() resizes
// to fit (Python's round(): halves to even), then the result is centered.
static Rgb pad(Rgb s, int bw, int bh) {
  double ir = (double)s.w / s.h, dr = (double)bw / bh;
  int w = bw, h = bh;
  if (ir > dr && (int)rint((double)s.h / s.w * bw) != bh) h = (int)rint((double)s.h / s.w * bw);
  if (ir < dr && (int)rint((double)s.w / s.h * bh) != bw) w = (int)rint((double)s.w / s.h * bh);
  Rgb r = resize(s, w, h);
  if (r.w == bw && r.h == bh) return r;
  Rgb o = {bw, bh, xmalloc((size_t)bw * bh * 3)};
  memset(o.p, 127, (size_t)bw * bh * 3);
  int x = r.w != bw ? (int)rint((bw - r.w) * 0.5) : 0, y = r.w != bw ? 0 : (int)rint((bh - r.h) * 0.5);
  for (int i = 0; i < r.h; i++) memcpy(o.p + ((size_t)(y + i) * bw + x) * 3, r.p + (size_t)i * r.w * 3, (size_t)r.w * 3);
  free(r.p);
  return o;
}

// ---- the image's patches, positions and tokens ----

uint8_t *image_decode_rgb(const void *data, size_t len, int *width, int *height, const char **err) {
  Rgb image = decode(data, len, err);
  *width = image.w, *height = image.h;
  return image.p;
}

uint8_t *image_resize_rgb(const uint8_t *pixels, int width, int height, int out_width, int out_height) {
  return resize((Rgb){width, height, (uint8_t *)pixels}, out_width, out_height).p;
}

uint8_t *image_rgb(const void *data, size_t len, int *bw, int *bh, int *th, int *tw, const char **err) {
  Rgb s = decode(data, len, err);
  if (!s.p) return 0;
  plan(s.w, s.h, bw, bh, th, tw);
  Rgb o = pad(s, *bw, *bh);
  free(s.p);
  return o.p;
}

static uint16_t bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

int image_open(Media *im, const void *data, size_t len, const char **err) {
  int bw, bh;
  uint8_t *rgb = image_rgb(data, len, &bw, &bh, &im->th, &im->tw, err);
  if (!rgb) return 0;
  im->vh = bh / PATCH, im->vw = bw / PATCH, im->rows = 0, im->start = 0;
  im->ntokens = tokens_of(im->th, im->tw), im->patch_stride = VIT_K, im->embedding_width = DIM;
  im->features = (Buf){0};
  im->hash = 1469598103934665603ull;  // FNV-1a of the file's bytes
  for (size_t i = 0; i < len; i++) im->hash = (im->hash ^ ((const uint8_t *)data)[i]) * 1099511628211ull;
  // the pixels as float32 in [-1, 1], rounded to BF16; a patch's channel c
  // at 196 c + 14 i + j (x.reshape(3, vh, 14, vw, 14).permute(1, 3, 0, 2, 4))
  size_t n = (size_t)im->vh * im->vw;
  im->patch = memory_host_alloc(n * VIT_K * 2, 0);
  if (!im->patch) { free(rgb); *err = "model-memory budget cannot fit the image patches"; return 0; }
  for (int y = 0; y < bh; y++)
    for (int x = 0; x < bw; x++)
      for (int c = 0; c < 3; c++) {
        float v = ((float)rgb[((size_t)y * bw + x) * 3 + c] / 255.0f - 0.5f) / 0.5f;
        size_t p = (size_t)(y / PATCH) * im->vw + x / PATCH;
        im->patch[p * VIT_K + c * PATCH * PATCH + (y % PATCH) * PATCH + x % PATCH] = bf16(v);
      }
  free(rgb);
  return 1;
}

// vision.py get_vision_cos_sin: patch (ph, pw) turns pairs 0-15 by ph and
// 16-31 by pw times the frequencies theta^(-2i/32), theta = 10000, all in
// float32 as the reference computes them.
void image_rope(const Media *im, float *cs) {
  float inv[16];
  for (int i = 0; i < 16; i++) inv[i] = 1.0f / powf(10000.0f, (float)(2 * i) / 32.0f);
  for (int ph = 0; ph < im->vh; ph++)
    for (int pw = 0; pw < im->vw; pw++) {
      float *row = cs + ((size_t)ph * im->vw + pw) * 64;
      for (int i = 0; i < 32; i++) {
        float a = (float)(i < 16 ? ph : pw) * inv[i % 16];
        row[i] = (float)cos((double)a), row[32 + i] = (float)sin((double)a);
      }
    }
}
