// Encode a bounded in-memory audio fixture and optionally transcribe it.
#include "../../src/model/model.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3)
    die("usage: speech-model CHECKPOINT [--features|--decode]");
  Str bytes = {0};
  char chunk[8192];
  ssize_t count;
  while ((count = read(0, chunk, sizeof chunk)) > 0) {
    if (bytes.n + count > 16u << 20)
      die("audio fixture exceeds 16 MiB");
    str_put(&bytes, chunk, count);
  }
  memory_configure(28, 4, 10);
  if (argc == 3 && !strcmp(argv[2], "--decode")) {
    Media audio = {0};
    const char *error;
    if (!audio_open(&audio, bytes.p, bytes.n, &error))
      die("%s", error);
    uint32_t header[2] = {(uint32_t)audio.sample_rate, (uint32_t)audio.sample_count};
    fwrite(header, sizeof header, 1, stdout);
    fwrite(audio.samples, 4, audio.sample_count, stdout);
    media_free(&audio);
    free(bytes.p);
    return 0;
  }
  Options options = {.ctx = 4096, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  Model *model = model_open(argv[1], &options);
  if (!model)
    die("model does not fit fixture budget");
  Media media = {0};
  const char *error;
  if (!model->ops->prepare_media(argv[1], &media, bytes.p, bytes.n, &error))
    die("%s", error);
  double began = now();
  if (!model->ops->encode_media(&media))
    die("speech encoder does not fit fixture budget");
  fprintf(stderr, "audio %.3f seconds: %d feature rows of %d BF16 values in %.3f seconds\n",
          (double)media.sample_count / media.sample_rate, media.ntokens, media.embedding_width, now() - began);
  if (argc == 3 && !strcmp(argv[2], "--features")) {
    const char *p = (const char *)media.rows;
    size_t size = (size_t)media.ntokens * media.embedding_width * 2;
    while (size) {
      ssize_t n = write(1, p, size);
      if (n <= 0)
        die("feature pipe closed");
      size -= n;
      p += n;
    }
  } else {
    Tokenizer *tokenizer = tok_open(argv[1], model->ops->media_token);
    char prompt[512];
    model->ops->chat(prompt, sizeof prompt, "Transcribe the following audio.", 1, 0);
    int ids[128], expanded[4096];
    int n = tok_encode_chat(tokenizer, prompt, ids, 128);
    n = media_expand(ids, n, &media, 1, expanded, 4096, tok_find(tokenizer, model->ops->media_token));
    if (n < 1)
      die("audio placeholder did not expand");
    model->ops->media(&media, 1);
    ModelState *state = model->ops->state_new();
    const float *logits = model->ops->prefill(state, expanded, n, 0, 0, -1, NULL, NULL);
    for (int i = 0; i < 512; i++) {
      int token = sample_argmax(logits, model->ops->vocab());
      if (model->ops->stop(token))
        break;
      int length;
      const char *piece = tok_piece(tokenizer, token, &length);
      fwrite(piece, 1, length, stdout);
      int position = n + i;
      logits = model->ops->rows(&state, &token, &position, 1);
    }
    puts("");
    model->ops->state_free(state);
    tok_close(tokenizer);
  }
  media_free(&media);
  free(bytes.p);
  return 0;
}
