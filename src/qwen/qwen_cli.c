// build/qwen: runs the Qwen3.8 engine (qwen.c) on token ids, for checks.
//
//   build/qwen --model DIR --tokens ID,ID,...|@FILE [--max N] [--check] [--ppl]
//
// Prefills the tokens in prompt passes of QWEN_PREFILL rows, decodes up to N
// tokens greedily (stopping at the model's end tokens), and prints the
// generated ids, their text (from the checkpoint's tokenizer.json) and the
// speeds. --check compares next-token logits bit for bit: the prompt in
// prompt passes of QWEN_PREFILL rows against passes of 37 rows, and in
// decoding passes of QWEN_ROWS rows against one row a pass. --ppl prints the
// prompt's
// perplexity (the mean negative log-likelihood of each token after the
// first, from the logits of every row). --rows N decodes N copies of the
// prompt's state together (one row each a pass) and prints the time a pass.
// --dump FILE writes the next-token logits after the prompt, from prompt
// passes and then from one-row decoding passes (2 x vocab floats), for
// comparing builds byte for byte. --mtp decodes greedily with the MTP head's
// drafts as well, and prints
// whether the tokens equal those decoded without drafts, the drafts
// accepted and the speeds.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../engine/myds4.h"
#include "qwen.h"

// The byte-level alphabet's code point for each byte (GPT-2's table).
static int byte_char[256];

static void alphabet(void) {
  int extra = 256;
  for (int b = 0; b < 256; b++)
    byte_char[b] = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) ? b : extra++;
}

// Appends a vocabulary piece's bytes: each UTF-8 code point is one byte of
// the alphabet.
static void piece_bytes(Str *out, const char *s, int n) {
  for (int i = 0; i < n;) {
    unsigned c = (unsigned char)s[i];
    int len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    unsigned cp = len == 1 ? c : c & (0x3F >> (len - 1));
    for (int k = 1; k < len && i + k < n; k++) cp = cp << 6 | ((unsigned char)s[i + k] & 0x3F);
    i += len;
    int byte = -1;
    for (int b = 0; b < 256; b++)
      if (byte_char[b] == (int)cp) byte = b;
    if (byte < 0) die("tokenizer.json: a piece outside the byte alphabet");
    char ch = (char)byte;
    str_put(out, &ch, 1);
  }
}

typedef struct { const char *s; int n, added; } VocabPiece;
static VocabPiece *pieces;
static int npieces;

static void load_vocab(const char *dir) {
  char path[4096];
  snprintf(path, sizeof path, "%s/tokenizer.json", dir);
  Json *root = json_parse(read_file(path, 0));
  const Json *vocab = json_get(json_get(root, "model"), "vocab"), *added = json_get(root, "added_tokens");
  if (!vocab || !added) die("%s: no model.vocab or added_tokens", path);
  npieces = qwen_vocab();
  pieces = calloc((size_t)npieces, sizeof *pieces);
  for (int i = 0; i < vocab->n; i++) {
    int id = (int)vocab->kid[i].num;
    if (id >= 0 && id < npieces) pieces[id] = (VocabPiece){vocab->kid[i].key, (int)strlen(vocab->kid[i].key), 0};
  }
  for (int i = 0; i < added->n; i++) {
    const Json *id = json_get(&added->kid[i], "id"), *content = json_get(&added->kid[i], "content");
    if (id && content && id->num >= 0 && id->num < npieces) pieces[(int)id->num] = (VocabPiece){content->s, content->n, 1};
  }
}

static void token_text(Str *out, int id) {
  if (id < 0 || id >= npieces || !pieces[id].s) return;
  if (pieces[id].added) str_put(out, pieces[id].s, (size_t)pieces[id].n);
  else piece_bytes(out, pieces[id].s, pieces[id].n);
}

static int *read_ids(const char *arg, int *n) {
  char *text = arg[0] == '@' ? read_file(arg + 1, 0) : (char *)arg;
  int *ids = xmalloc((strlen(text) / 2 + 2) * sizeof *ids);
  *n = 0;
  for (char *p = text; *p;) {
    while (*p == ',' || *p == ' ' || *p == '\n') p++;
    if (!*p) break;
    char *end;
    ids[(*n)++] = (int)strtol(p, &end, 10);
    if (end == p) die("--tokens: not a list of ids");
    p = end;
  }
  return ids;
}

// Runs tokens [0, n) of a state per_pass at a time, in prompt passes or in
// decoding passes; returns the last row's logits, and with nll adds each next
// token's negative log-likelihood (decoding passes: logits for every row).
static const float *prefill(QwenState *s, const int *tokens, int n, int per_pass, int prompt, double *nll) {
  const float *logits = 0;
  static QwenRow rows[QWEN_PREFILL];
  for (int i = 0; i < n; i += per_pass) {
    int k = n - i < per_pass ? n - i : per_pass;
    for (int j = 0; j < k; j++) rows[j] = (QwenRow){s, tokens[i + j], 0};
    logits = prompt ? qwen_prefill(rows, k, 1) : qwen_forward(rows, k, nll ? k : 1);
    for (int j = 0; nll && j < k && i + j + 1 < n; j++) {
      const float *l = logits + (size_t)j * qwen_vocab();
      double m = -INFINITY, sum = 0;
      for (int v = 0; v < qwen_vocab(); v++) m = l[v] > m ? l[v] : m;
      for (int v = 0; v < qwen_vocab(); v++) sum += exp(l[v] - m);
      *nll += m + log(sum) - l[tokens[i + j + 1]];
    }
    if (nll) logits += (size_t)(k - 1) * qwen_vocab();
  }
  return logits;
}

int main(int argc, char **argv) {
  const char *dir = 0, *list = 0;
  int max = 64, check = 0, ppl = 0, nrows = 0, use_mtp = 0;
  const char *dump = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
    else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) list = argv[++i];
    else if (!strcmp(argv[i], "--max") && i + 1 < argc) max = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--check")) check = 1;
    else if (!strcmp(argv[i], "--ppl")) ppl = 1;
    else if (!strcmp(argv[i], "--rows") && i + 1 < argc) nrows = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--mtp")) use_mtp = 1;
    else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
    else die("usage: qwen --model DIR --tokens ID,ID,...|@FILE [--max N] [--check] [--ppl]");
  }
  if (!dir || !list) die("usage: qwen --model DIR --tokens ID,ID,...|@FILE [--max N] [--check] [--ppl]");
  int n;
  int *prompt = read_ids(list, &n);
  if (!n) die("--tokens: no ids");
  alphabet();
  qwen_load(dir);
  load_vocab(dir);
  QwenState *s = qwen_state_new();
  qwen_state_reserve(s, n + max);
  double t0 = now(), nll = 0;
  const float *logits = ppl ? prefill(s, prompt, n, QWEN_ROWS, 0, &nll) : prefill(s, prompt, n, QWEN_PREFILL, 1, 0);
  double t1 = now();
  if (dump) {
    FILE *f = fopen(dump, "wb");
    if (!f) die("cannot write %s", dump);
    fwrite(logits, sizeof *logits, (size_t)qwen_vocab(), f);
    QwenState *one = qwen_state_new();
    qwen_state_reserve(one, n);
    const float *l1 = prefill(one, prompt, n, 1, 0, 0);
    fwrite(l1, sizeof *l1, (size_t)qwen_vocab(), f);
    fclose(f);
    qwen_state_free(one);
    printf("dump: %s\n", dump);
    return 0;
  }
  if (ppl) printf("perplexity: %.4f over %d tokens (mean NLL %.4f)\n", exp(nll / (n - 1)), n - 1, nll / (n - 1));
  if (check) {
    struct { int per_pass, prompt; const char *what; } runs[4] = {
        {QWEN_PREFILL, 1, "prompt passes of QWEN_PREFILL rows"}, {37, 1, "prompt passes of 37 rows"},
        {QWEN_ROWS, 0, "decoding passes of QWEN_ROWS rows"}, {1, 0, "decoding passes of one row"}};
    float *got[4];
    int bad = 0;
    for (int r = 0; r < 4; r++) {
      QwenState *t = qwen_state_new();
      qwen_state_reserve(t, n);
      const float *l = prefill(t, prompt, n, runs[r].per_pass, runs[r].prompt, 0);
      got[r] = xmalloc((size_t)qwen_vocab() * sizeof *got[r]);
      memcpy(got[r], l, (size_t)qwen_vocab() * sizeof *got[r]);
      qwen_state_free(t);
    }
    for (int r = 0; r < 4; r += 2) {
      int differ = 0;
      double worst = 0;
      for (int v = 0; v < qwen_vocab(); v++) {
        if (memcmp(&got[r][v], &got[r + 1][v], 4)) differ++;
        double d = fabs((double)got[r][v] - got[r + 1][v]);
        worst = d > worst ? d : worst;
      }
      printf("check: %d of %d logits differ between %s and %s (largest difference %g); argmax %d and %d\n", differ,
             qwen_vocab(), runs[r].what, runs[r + 1].what, worst, sample_argmax(got[r], qwen_vocab()),
             sample_argmax(got[r + 1], qwen_vocab()));
      bad |= differ != 0;
    }
    double worst = 0;
    for (int v = 0; v < qwen_vocab(); v++) {
      double d = fabs((double)got[0][v] - got[3][v]);
      worst = d > worst ? d : worst;
    }
    printf("check: prompt passes against decoding passes: largest difference %g, argmax %d and %d\n", worst,
           sample_argmax(got[0], qwen_vocab()), sample_argmax(got[3], qwen_vocab()));
    if (bad) return 1;
  }
  if (nrows > 0) {  // decode nrows copies of the prompt's state together
    if (nrows > QWEN_ROWS) die("--rows: at most %d", QWEN_ROWS);
    QwenState *st[QWEN_ROWS];
    int next[QWEN_ROWS];
    for (int r = 0; r < nrows; r++) {
      st[r] = qwen_state_new();
      qwen_state_copy(st[r], s, n);
      qwen_state_reserve(st[r], n + max + 3);
      next[r] = sample_argmax(logits, qwen_vocab());
    }
    QwenRow rows[QWEN_ROWS];
    double a = 0;
    for (int step = -2; step < max; step++) {  // two passes to warm up
      if (step == 0) a = now();
      for (int r = 0; r < nrows; r++) rows[r] = (QwenRow){st[r], next[r], 0};
      const float *l = qwen_forward(rows, nrows, nrows);
      for (int r = 0; r < nrows; r++) next[r] = sample_argmax(l + (size_t)r * qwen_vocab(), qwen_vocab());
    }
    double b = now();
    printf("rows %d: %.2f ms a pass, %.1f tokens/s together, %.1f each\n", nrows, (b - a) / max * 1e3,
           nrows * max / (b - a), max / (b - a));
    return 0;
  }
  Str text = {0};
  printf("generated:");
  int made = 0, steps = 0;
  double t2 = now();
  while (made < max) {
    int next = sample_argmax(logits, qwen_vocab());
    made++;
    printf(" %d", next);
    fflush(stdout);
    if (qwen_stop(next)) break;
    token_text(&text, next);
    if (made == max) break;
    QwenRow row = {s, next, 0};
    logits = qwen_forward(&row, 1, 1);
    steps++;
  }
  double t3 = now();
  if (use_mtp) {  // the same prompt again, with drafts
    QwenState *m = qwen_state_new();
    qwen_state_reserve(m, n + max + 3);
    const float *l = prefill(m, prompt, n, QWEN_PREFILL, 1, 0);
    int y = sample_argmax(l, qwen_vocab()), made = 0, passes = 0, accepted = 0, drafted = 0, stop = 0;
    int *out = xmalloc((size_t)(max + 2) * sizeof *out);
    double a = now();
    out[made++] = y;
    stop = qwen_stop(y);
    QwenMtpRow mr = {m, y, 0};
    int d = sample_argmax(qwen_mtp(&mr, 1, 1), qwen_vocab());
    while (!stop && made < max) {
      QwenRow rows[2] = {{m, y, 1}, {m, d, 0}};
      const float *v = qwen_forward(rows, 2, 2);
      passes++, drafted++;
      int y2 = sample_argmax(v, qwen_vocab());
      out[made++] = y2;
      if (qwen_stop(y2) || made >= max) break;
      if (y2 == d) {
        accepted++;
        int y3 = sample_argmax(v + qwen_vocab(), qwen_vocab());
        out[made++] = y3;
        if (qwen_stop(y3) || made >= max) break;
        QwenMtpRow two[2] = {{m, y2, 1}, {m, y3, 0}};
        d = sample_argmax(qwen_mtp(two, 2, 1), qwen_vocab());
        y = y3;
      } else {
        qwen_state_back(m);
        QwenMtpRow one = {m, y2, 0};
        d = sample_argmax(qwen_mtp(&one, 1, 1), qwen_vocab());
        y = y2;
      }
    }
    double b = now();
    // the tokens decoded without drafts: the run above (generated ids printed), re-decoded here
    QwenState *g = qwen_state_new();
    qwen_state_reserve(g, n + max + 3);
    l = prefill(g, prompt, n, QWEN_PREFILL, 1, 0);
    int same = 1, k = 0;
    for (; k < made; k++) {
      int t = sample_argmax(l, qwen_vocab());
      if (t != out[k]) {
        same = 0;
        break;
      }
      if (k + 1 < made) {
        QwenRow r = {g, t, 0};
        l = qwen_forward(&r, 1, 1);
      }
    }
    printf("\nmtp: %d tokens in %d passes, %d of %d drafts accepted (%.0f%%), %.2f tokens/s; %s\n", made, passes,
           accepted, drafted, drafted ? 100.0 * accepted / drafted : 0.0, (made - 1) / (b - a),
           same ? "the same tokens as without drafts" : "DIFFERENT tokens from those without drafts");
    if (!same) printf("mtp: first difference at token %d\n", k);
  }
  printf("\ntext: %s\n", text.p ? text.p : "");
  printf("prefill: %d tokens in %.2f s (%.1f tok/s); decode: %d steps in %.2f s (%.2f tok/s)\n", n, t1 - t0,
         n / (t1 - t0), steps, t3 - t2, steps / (t3 - t2 > 0 ? t3 - t2 : 1));
  return 0;
}
