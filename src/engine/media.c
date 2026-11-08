// Encoded media spans and their feature buffers.
#include "myds4.h"
#include "memory.h"

static uint64_t mix(uint64_t x) { // splitmix64's finalizer
  x ^= x >> 30, x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27, x *= 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

int media_span(const Media *im) {
  return im->ntokens;
}

uint64_t media_embedding(const Media *media, int count, int position) {
  for (int i = 0; i < count; i++) {
    int row = position - media[i].start;
    if (row >= 0 && row < media[i].ntokens)
      return media[i].features.gpu + (size_t)row * media[i].embedding_width * 2;
  }
  die("no media embedding at position %d", position);
}

int media_expand(const int *in, int n, Media *im, int nim, int *out, int max, int media_token) {
  int k = 0, m = 0;
  for (int i = 0; i < n; i++) {
    if (in[i] != media_token) {
      if (m < max)
        out[m] = in[i];
      m++;
      continue;
    }
    if (k == nim)
      return -1;
    int s = media_span(&im[k]);
    im[k].start = m;
    if (m + s <= max)
      media_keys(&im[k], out + m);
    m += s, k++;
  }
  return k == nim ? m : -1;
}

void media_keys(const Media *im, int *keys) {
  for (int j = 0, n = media_span(im); j < n; j++)
    keys[j] = -1 - (int)(mix(im->hash + (uint64_t)j) & 0x3fffffff);
}

void media_free(Media *im) {
  if (im->kind == MEDIA_AUDIO)
    memory_host_free(im->samples);
  else
    memory_host_free(im->patch);
  if (im->features.obj)
    gpu_free(&im->features);
  else
    memory_host_free(im->rows);
  *im = (Media){0};
}
