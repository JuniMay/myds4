// myds4's inference engine behind a C ABI, for a server written in another
// language (server/, in Rust). The engine owns the model and everything that
// runs on it: loading, the conversations' attention state (sessions), the
// request queue and the batching scheduler, sampling, DSpark drafts, the
// vision encoder, the live view (--tui), the log and the stop on SIGINT,
// SIGTERM or the view's q key. The caller owns HTTP, request conversion,
// prompt rendering and output parsing.
//
// One process runs one engine. The engine runs the model on a thread of its
// own, which loads the model and serves the queue; every function below may
// be called from any thread. Event callbacks run on the engine's thread: they
// must return quickly, must not wait for another thread, and may call only
// myds4_cancel and myds4_log.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define MYDS4_ABI 1

// The engine's settings: the C server's model and server options.
typedef struct {
  int abi;                     // MYDS4_ABI
  const char *model;           // -m: the checkpoint directory
  const char *experts;         // --experts: a GGUF file or any shard of a set; null: the checkpoint's
  int ctx;                     // --ctx: context capacity in tokens, 1 to 1,048,576
  double cache_gb;             // --cache: expert cache budget in GB; 0: from available memory
  int cache_half_life;         // --cache-half-life in forward passes; 0: the default, 50
  int cache_read_cost;         // --cache-read-cost
  float substitute;            // --substitute D; 0: exact
  int gpu_route;               // --gpu-route
  int draft;                   // --draft N, 0 to 5
  int draft_cache;             // --draft-cache N; 0: all draft experts preloaded
  int draft_trim;              // --draft-trim
  int online_predictor;        // --online-predictor
  const char *predictor_state; // --predictor-state FILE, or null
  double kv_gb;                // --kv-cache: bound of the sessions' attention state in GiB
  int batch;                   // --batch: requests decoding together, 1 to 6
  int tui;                     // --tui: the live view on stderr instead of the log
  const char *address;         // "host:port", for the view's title
} myds4_config;

// Waits for the lock that one model process holds at a time (myds4.lock
// beside the executable, or $MYDS4_LOCK), starts the engine's thread, loads
// the model and installs the stop's signal handlers; returns once the engine
// takes requests. 0, or -1 with a message on stderr for invalid settings. A
// model that cannot be loaded ends the process, as in the C server.
int myds4_start(const myds4_config *c);

// Whether a stop was requested: SIGINT, SIGTERM, the live view's q, or
// myds4_stop. The engine then ends every request with MYDS4_SHUTDOWN.
int myds4_stopping(void);

// Requests the stop, waits for the engine's thread to end every request,
// saves the predictor state (--predictor-state) and returns.
void myds4_stop(void);

// ---- requests ----

typedef struct {
  const uint8_t *data;  // an image file's bytes, in any format macOS reads
  size_t len;
} myds4_image;

enum { MYDS4_STARTED, MYDS4_TOKENS, MYDS4_DONE, MYDS4_REFUSED };  // myds4_event.kind
enum { MYDS4_STOP, MYDS4_LENGTH, MYDS4_CANCELLED, MYDS4_SHUTDOWN };  // myds4_event.finish

typedef struct {
  int kind;
  uint64_t id;
  // MYDS4_TOKENS: the tokens a decoding step kept, in order (several with
  // drafts); the end token is never among them. Valid during the callback.
  const int32_t *tokens;
  int ntokens;
  int prompt, reused;     // STARTED and DONE: prompt tokens (images expanded), those a session held
  int generated;          // DONE: generated tokens, the end token counted
  int reasoning;          // DONE: of them, those up to and including </think> when the prompt ends in <think>
  int finish;             // DONE: MYDS4_STOP (the end token), MYDS4_LENGTH (max_tokens or the
                          // context's end), MYDS4_CANCELLED (myds4_cancel), MYDS4_SHUTDOWN
  const char *message;    // REFUSED: why, for the client. Valid during the callback.
} myds4_event;

// One request: its whole prompt, prefilled in one pass, and its sampling.
// The engine copies what it needs before myds4_submit returns.
typedef struct {
  // The prompt as token ids, or as rendered text (when tokens is null) that
  // the engine tokenizes as the C server does: a special token's spelling is
  // that token, and the byte IMAGE_MARK (0xFF) an image's place.
  const int32_t *tokens;
  int ntokens;
  const char *text;
  size_t len;
  // The images of the prompt's image places (IMAGE_TOKEN 129264), in order.
  const myds4_image *images;
  int nimages;
  double temperature;  // finite and nonnegative; 0 is greedy
  double top_p;        // 0 to 1
  uint64_t seed;
  int seed_set;        // 0: a random seed
  long max_tokens;     // at least 1; the context's end stops the request too
  // Events, in this order: STARTED once its prefill ran, TOKENS after each
  // step that kept tokens, then exactly one DONE (without STARTED for a
  // request cancelled or stopped while it waited); or REFUSED alone, before
  // its prefill (the context or --kv-cache cannot hold it, or an image
  // cannot be read). After DONE or REFUSED the engine no longer uses user.
  void (*on_event)(void *user, const myds4_event *e);
  void *user;
  // Logged as "req ID NOTE" when the request is queued, before any line of
  // its prefill (for example where it came from); or null.
  const char *note;
} myds4_request;

enum { MYDS4_QUEUED, MYDS4_BUSY, MYDS4_INVALID };  // myds4_submit's results

// Queues a request: MYDS4_QUEUED with its id (the log's request number,
// never 0); MYDS4_BUSY when 16 requests are waiting; MYDS4_INVALID with a
// message for invalid sampling settings or a missing prompt or callback. Only
// MYDS4_QUEUED gives events. The worker admits waiting requests in arrival
// order while fewer than --batch decode; a queued prompt is never split.
int myds4_submit(const myds4_request *r, uint64_t *id, char *message, size_t cap);

// Ends a request: a waiting one is dropped, a running one stops before its
// next step; either ends with DONE, MYDS4_CANCELLED. Unknown or finished ids
// are ignored.
void myds4_cancel(uint64_t id);

// ---- the server's state and log ----

// What the engine is doing, as GET /v1/status answers it (JSON); allocated,
// free it with myds4_free.
char *myds4_status(void);
void myds4_free(void *p);

// A line for the log: stderr with the time, or the live view's log pane.
void myds4_log(const char *line);

// ---- the checkpoint's tokenizer (tokenizer.json) ----

// Loads it without the model (myds4_start loads it too) and returns 0; a
// tokenizer.json that cannot be read ends the process with a message.
int myds4_tokenizer(const char *dir);
// The token ids of text: chat 1 as a submitted prompt's text, chat 0 as
// plain text without special tokens. Returns the count, or -1 if more than
// max.
int myds4_tokenize(const char *text, size_t len, int chat, int32_t *out, int max);
// A token's bytes (len 0 for an unknown id), and whether it is one of the
// tokenizer's added tokens (the special tokens among them).
const char *myds4_piece(int32_t id, size_t *len);
int myds4_added(int32_t id);
