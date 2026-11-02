// Chat Completions requests as DeepSeek-V4.1 prompts, after the checkpoint's
// encoding/encoding.py (text only), and the model's output back as
// reasoning, content and tool calls.
#include <stdlib.h>
#include <string.h>
#include <yyjson.h>

#include "myds4.h"

#define EOS "<｜end▁of▁sentence｜>"
#define CALLS "<｜DSML｜ calls"

// ---- JSON as Python's json.dumps(v, ensure_ascii=False) writes it ----

void json_str(Str *o, const char *s, size_t n) {
  str_put(o, "\"", 1);
  for (size_t i = 0; i < n; i++) {
    unsigned char c = s[i];
    const char *e = c == '"' ? "\\\"" : c == '\\' ? "\\\\" : c == '\n' ? "\\n" : c == '\r' ? "\\r"
                  : c == '\t' ? "\\t" : c == '\b' ? "\\b" : c == '\f' ? "\\f" : 0;
    if (e) str_cat(o, e);
    else if (c < 0x20) str_printf(o, "\\u%04x", c);
    else str_put(o, (const char *)&s[i], 1);
  }
  str_put(o, "\"", 1);
}

// Separators ", " and ": "; numbers as the request wrote them (read as raw text).
void json_val(Str *o, yyjson_val *v) {
  size_t i, max;
  yyjson_val *k, *x;
  switch (yyjson_get_type(v)) {
  case YYJSON_TYPE_BOOL: str_cat(o, yyjson_get_bool(v) ? "true" : "false"); break;
  case YYJSON_TYPE_RAW: str_cat(o, yyjson_get_raw(v)); break;
  case YYJSON_TYPE_NUM: str_printf(o, "%.17g", yyjson_get_num(v)); break;
  case YYJSON_TYPE_STR: json_str(o, yyjson_get_str(v), yyjson_get_len(v)); break;
  case YYJSON_TYPE_ARR:
    str_cat(o, "[");
    yyjson_arr_foreach(v, i, max, x) {
      if (i) str_cat(o, ", ");
      json_val(o, x);
    }
    str_cat(o, "]");
    break;
  case YYJSON_TYPE_OBJ:
    str_cat(o, "{");
    yyjson_obj_foreach(v, i, max, k, x) {
      if (i) str_cat(o, ", ");
      json_str(o, yyjson_get_str(k), yyjson_get_len(k));
      str_cat(o, ": ");
      json_val(o, x);
    }
    str_cat(o, "}");
    break;
  default: str_cat(o, "null");
  }
}

// ---- messages ----

enum { SYSTEM, USER, ASSISTANT, REMINDER };
enum { TEXT, IMAGE, RESULT, OTHER };

typedef struct { int type; yyjson_val *v; const char *id; } Block;  // v: the text, the result or the type
typedef struct {
  int role;
  yyjson_val *m;   // the message (null for an inserted system message)
  Block *b;        // a user turn: its parts and the tool results merged into it
  int nb, cap;
  yyjson_val *tools;
} Msg;

static void add(Msg *m, Block b) {
  if (m->nb == m->cap) m->b = realloc(m->b, (m->cap = m->cap * 2 + 4) * sizeof *m->b);
  m->b[m->nb++] = b;
}

static const char *str_of(yyjson_val *o, const char *key) { return yyjson_get_str(yyjson_obj_get(o, key)); }

static int is_image(const char *type) { return type && (!strcmp(type, "image_url") || !strcmp(type, "image")); }

// The image parts in the order chat_prompt writes their placeholders.
static yyjson_val **img_parts;
static int img_count, img_cap;

static void put_image(Str *o, yyjson_val *part, const char *image) {
  str_cat(o, image);
  if (img_count == img_cap) img_parts = realloc(img_parts, (img_cap = img_cap * 2 + 4) * sizeof *img_parts);
  img_parts[img_count++] = part;
}

// A content array's parts as blocks: text, images and the rest.
static void add_parts(Msg *m, yyjson_val *content) {
  size_t i, max;
  yyjson_val *p;
  if (!yyjson_is_arr(content)) {
    add(m, (Block){.type = TEXT, .v = content});
    return;
  }
  yyjson_arr_foreach(content, i, max, p) {
    const char *type = str_of(p, "type");
    if (type && !strcmp(type, "text")) add(m, (Block){.type = TEXT, .v = yyjson_obj_get(p, "text")});
    else if (is_image(type)) add(m, (Block){.type = IMAGE, .v = p});
    else add(m, (Block){.type = OTHER, .v = yyjson_obj_get(p, "type")});
  }
}

// Text content: a string, or the text parts of an array joined by blank lines.
static void put_text(Str *o, yyjson_val *content, const char *image) {
  size_t i, max, n = 0;
  yyjson_val *p;
  if (yyjson_is_str(content)) str_put(o, yyjson_get_str(content), yyjson_get_len(content));
  yyjson_arr_foreach(content, i, max, p) {
    const char *type = str_of(p, "type");
    int text = type && !strcmp(type, "text");
    if (!text && !is_image(type)) continue;
    if (n++) str_cat(o, "\n\n");
    if (text) put_text(o, yyjson_obj_get(p, "text"), image);
    else put_image(o, p, image);
  }
}

static void put_unsupported(Str *o, yyjson_val *type) {
  str_cat(o, "[Unsupported ");
  str_cat(o, yyjson_get_str(type) ? yyjson_get_str(type) : "None");
  str_cat(o, "]");
}

static void put_block(Str *o, const Block *b, const char *image) {
  size_t i, max;
  yyjson_val *p;
  switch (b->type) {
  case TEXT: put_text(o, b->v, image); break;
  case IMAGE: put_image(o, b->v, image); break;
  case OTHER: put_unsupported(o, b->v); break;
  case RESULT:
    str_cat(o, "<tool_result>");
    if (!yyjson_is_arr(b->v)) put_text(o, b->v, image);
    yyjson_arr_foreach(b->v, i, max, p) {
      const char *type = str_of(p, "type");
      if (i) str_cat(o, "\n\n");
      if (type && !strcmp(type, "text")) put_text(o, yyjson_obj_get(p, "text"), image);
      else if (is_image(type)) put_image(o, p, image);
      else put_unsupported(o, yyjson_obj_get(p, "type"));
    }
    str_cat(o, "</tool_result>");
  }
}

// The request's messages as encoding.py prepares them: tool messages become
// tool results of a user turn, consecutive user messages merge, and a turn's
// tool results follow the order of the calls before them. The request's
// tools belong to the first message, a system message (inserted if needed).
static Msg *load(yyjson_val *messages, yyjson_val *tools, int *count, const char **err) {
  size_t i, max, n = 0;
  Msg *ms = xmalloc((yyjson_arr_size(messages) + 2) * sizeof *ms);
  yyjson_val *m;
  if (!yyjson_arr_size(messages)) *err = "messages must be a non-empty array";
  if (yyjson_arr_size(tools)) ms[n++] = (Msg){.role = SYSTEM, .tools = tools};
  yyjson_arr_foreach(messages, i, max, m) {
    const char *role = str_of(m, "role");
    yyjson_val *content = yyjson_obj_get(m, "content");
    if (!role) {
      *err = "a message has no role";
    } else if (!strcmp(role, "tool")) {
      if (!n || ms[n - 1].role != USER) ms[n++] = (Msg){.role = USER, .m = m};
      add(&ms[n - 1], (Block){.type = RESULT, .v = content, .id = str_of(m, "tool_call_id")});
    } else if (!strcmp(role, "user")) {
      if (!n || ms[n - 1].role != USER) ms[n++] = (Msg){.role = USER, .m = m};
      add_parts(&ms[n - 1], content);
    } else if (!strcmp(role, "system") || !strcmp(role, "developer")) {
      if (n == 1 && i == 0 && ms[0].tools && !ms[0].m) n = 0;  // the request's first message takes its tools
      ms[n] = (Msg){.role = SYSTEM, .m = m, .tools = n == 0 && yyjson_arr_size(tools) ? tools : 0};
      if (!ms[n].tools && yyjson_arr_size(yyjson_obj_get(m, "tools"))) ms[n].tools = yyjson_obj_get(m, "tools");
      n++;
    } else if (!strcmp(role, "assistant")) {
      ms[n++] = (Msg){.role = ASSISTANT, .m = m};
    } else if (!strcmp(role, "latest_reminder")) {
      ms[n++] = (Msg){.role = REMINDER, .m = m};
    } else {
      *err = "a message has an unknown role";
    }
  }
  yyjson_val *calls = 0;  // the tool calls of the last assistant message
  for (size_t k = 0; k < n; k++) {
    if (ms[k].role == ASSISTANT && yyjson_arr_size(yyjson_obj_get(ms[k].m, "tool_calls")))
      calls = yyjson_obj_get(ms[k].m, "tool_calls");
    if (ms[k].role != USER || !calls || ms[k].nb < 2) continue;
    int rank[ms[k].nb], at[ms[k].nb], nr = 0;
    for (int b = 0; b < ms[k].nb; b++) {
      if (ms[k].b[b].type != RESULT) continue;
      size_t c, cmax;
      yyjson_val *call;
      rank[nr] = 0, at[nr++] = b;
      yyjson_arr_foreach(calls, c, cmax, call) {
        const char *id = str_of(call, "id");
        if (id && ms[k].b[b].id && !strcmp(id, ms[k].b[b].id)) rank[nr - 1] = (int)c;
      }
    }
    if (nr < 2) continue;
    Block sorted[nr];
    int order[nr];
    for (int a = 0; a < nr; a++) order[a] = a;
    for (int a = 1; a < nr; a++)  // stable, as Python's sorted
      for (int b = a; b > 0 && rank[order[b - 1]] > rank[order[b]]; b--) {
        int t = order[b]; order[b] = order[b - 1], order[b - 1] = t;
      }
    for (int a = 0; a < nr; a++) sorted[a] = ms[k].b[at[order[a]]];
    for (int a = 0; a < nr; a++) ms[k].b[at[a]] = sorted[a];
  }
  *count = (int)n;
  return ms;
}

// The last message that starts a turn: a user message with text or images,
// or a system message after the first.
static int last_user(const Msg *ms, int n) {
  for (int i = n - 1; i >= 0; i--) {
    if (ms[i].role == SYSTEM && i > 0) return i;
    if (ms[i].role != USER) continue;
    int text = !ms[i].nb;
    for (int b = 0; b < ms[i].nb; b++) text |= ms[i].b[b].type == TEXT || ms[i].b[b].type == IMAGE;
    if (text) return i;
  }
  return -1;
}

static void put_tools(Str *o, yyjson_val *tools) {
  size_t i, max, n = 0;
  yyjson_val *t;
  str_cat(o, "## Tools\n\n"
             "You have access to a set of tools to help answer the user's question. You can invoke tools by "
             "writing a \"<｜DSML｜ calls>\" block like the following:\n\n"
             "<｜DSML｜ calls>\n"
             "<｜DSML｜ invoke name=\"$TOOL_NAME\">\n"
             "<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n"
             "...\n"
             "</｜DSML｜ invoke>\n"
             "<｜DSML｜ invoke name=\"$TOOL_NAME2\">\n"
             "...\n"
             "</｜DSML｜ invoke>\n"
             "</｜DSML｜ calls>\n\n"
             "String parameters should be specified as is and set `string=\"true\"`. For all other types "
             "(numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
             "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning "
             "inside <think>...</think> BEFORE any tool calls or final response.\n\n"
             "Otherwise, output directly after </think> with tool calls or final response.\n\n"
             "### Available Tool Schemas\n\n");
  yyjson_arr_foreach(tools, i, max, t) {
    yyjson_val *f = yyjson_obj_get(t, "function");
    if (!yyjson_is_obj(f)) continue;
    if (n++) str_cat(o, "\n");
    json_val(o, f);
  }
  str_cat(o, "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n");
}

// A call's arguments as DSML parameters: strings as they are, other values as JSON.
static void put_arguments(Str *o, yyjson_val *args) {
  yyjson_doc *doc[2] = {0};
  yyjson_val *a = args, *k, *v;
  for (int d = 0; d < 2 && yyjson_is_str(a); d++) {
    doc[d] = yyjson_read(yyjson_get_str(a), yyjson_get_len(a), YYJSON_READ_NUMBER_AS_RAW);
    if (!doc[d]) break;
    a = yyjson_doc_get_root(doc[d]);
  }
  size_t i, max;
  if (yyjson_is_obj(a)) {
    yyjson_obj_foreach(a, i, max, k, v) {
      if (i) str_cat(o, "\n");
      str_printf(o, "<｜DSML｜ parameter name=\"%s\" string=\"%s\">", yyjson_get_str(k),
                 yyjson_is_str(v) ? "true" : "false");
      if (yyjson_is_str(v)) str_put(o, yyjson_get_str(v), yyjson_get_len(v));
      else json_val(o, v);
      str_cat(o, "</｜DSML｜ parameter>");
    }
  } else {  // not an object: one parameter named arguments
    str_printf(o, "<｜DSML｜ parameter name=\"arguments\" string=\"%s\">", yyjson_is_str(args) ? "true" : "false");
    if (yyjson_is_str(args)) str_put(o, yyjson_get_str(args), yyjson_get_len(args));
    else json_val(o, args);
    str_cat(o, "</｜DSML｜ parameter>");
  }
  yyjson_doc_free(doc[0]), yyjson_doc_free(doc[1]);
}

char *chat_prompt(yyjson_val *messages, yyjson_val *tools, int thinking, int effort, const char *image,
                  const char **err, yyjson_val ***images, int *nimages) {
  int n;
  *err = 0, img_count = 0;
  Msg *ms = load(messages, tools, &n, err);
  if (*err) {
    for (int i = 0; i < n; i++) free(ms[i].b);
    free(ms);
    return 0;
  }
  int drop = 1, last = last_user(ms, n);  // earlier turns' reasoning is dropped unless there are tools
  for (int i = 0; i < n; i++) drop &= !ms[i].tools;
  Str o = {0};
  str_cat(&o, "<｜begin▁of▁sentence｜>");
  for (int i = 0; i < n; i++) {
    Msg *m = &ms[i];
    yyjson_val *content = m->m ? yyjson_obj_get(m->m, "content") : 0;
    if (i == 0 && (thinking || m->role == SYSTEM)) str_cat(&o, "<｜System｜>");
    if (i == 0 && thinking)
      str_printf(&o, "Reasoning Effort: %d (range 1-100, the higher the value, the more thorough the reasoning)\n\n",
                 effort);
    switch (m->role) {
    case SYSTEM:
      if (i > 0) str_cat(&o, "<｜System｜>");
      put_text(&o, content, image);
      if (m->tools) str_cat(&o, "\n\n"), put_tools(&o, m->tools);
      break;
    case USER:
      str_cat(&o, "<｜User｜>");
      for (int b = 0; b < m->nb; b++) {
        if (b) str_cat(&o, "\n\n");
        put_block(&o, &m->b[b], image);
      }
      break;
    case REMINDER:
      str_cat(&o, "<｜latest_reminder｜>");
      put_text(&o, content, image);
      break;
    case ASSISTANT: {
      if (thinking && (!drop || i > last)) {
        yyjson_val *r = yyjson_obj_get(m->m, "reasoning_content");
        put_text(&o, r ? r : yyjson_obj_get(m->m, "reasoning"), image);
        str_cat(&o, "</think>");
      }
      put_text(&o, content, image);
      yyjson_val *calls = yyjson_obj_get(m->m, "tool_calls"), *c;
      size_t k, max;
      if (yyjson_arr_size(calls)) str_cat(&o, "\n\n<｜DSML｜ calls>\n");
      yyjson_arr_foreach(calls, k, max, c) {
        yyjson_val *f = yyjson_obj_get(c, "function");
        if (k) str_cat(&o, "\n");
        str_printf(&o, "<｜DSML｜ invoke name=\"%s\">\n", str_of(f, "name") ? str_of(f, "name") : "");
        put_arguments(&o, yyjson_obj_get(f, "arguments"));
        str_cat(&o, "\n</｜DSML｜ invoke>");
      }
      if (yyjson_arr_size(calls)) str_cat(&o, "\n</｜DSML｜ calls>");
      str_cat(&o, EOS);
      break;
    }
    }
    if (i + 1 < n && ms[i + 1].role != ASSISTANT && ms[i + 1].role != REMINDER) continue;
    if (m->role == USER || (m->role == SYSTEM && i > 0))
      str_cat(&o, thinking && (!drop || i >= last) ? "<｜Assistant｜><think>" : "<｜Assistant｜></think>");
  }
  for (int i = 0; i < n; i++) free(ms[i].b);
  free(ms);
  if (images) {
    *nimages = img_count;
    *images = xmalloc((img_count + 1) * sizeof **images);
    memcpy(*images, img_parts, img_count * sizeof **images);
  }
  return o.p;
}

// A number 1-100, or a level as DeepSeek's request converter (deepseek-recipe)
// maps it for V4.1: minimal and low 50, medium, high and xhigh 75, ultra and
// max 100.
int chat_effort(yyjson_val *v, int dflt) {
  const char *s = yyjson_get_str(v);
  int e = yyjson_is_raw(v) ? atoi(yyjson_get_raw(v)) : s ? atoi(s) : 0;
  if (s && (!strcmp(s, "minimal") || !strcmp(s, "low"))) e = 50;
  if (s && (!strcmp(s, "medium") || !strcmp(s, "high") || !strcmp(s, "xhigh"))) e = 75;
  if (s && (!strcmp(s, "ultra") || !strcmp(s, "max"))) e = 100;
  return e >= 1 && e <= 100 ? e : dflt;
}

// --render: a test case of encoding/tests (a message list, or an object with
// messages, tools, thinking_mode and reasoning_effort) as its prompt; on
// stderr, each image part's URL in the order of its placeholder.
void chat_render_file(const char *path) {
  size_t len;
  char *text = read_file(path, &len);
  yyjson_doc *doc = yyjson_read(text, len, YYJSON_READ_NUMBER_AS_RAW);
  if (!doc) die("%s: not JSON", path);
  yyjson_val *root = yyjson_doc_get_root(doc), *msgs = yyjson_is_arr(root) ? root : yyjson_obj_get(root, "messages");
  const char *mode = str_of(root, "thinking_mode"), *err;
  yyjson_val **images;
  int nimages;
  char *prompt = chat_prompt(msgs, yyjson_obj_get(root, "tools"), mode && !strcmp(mode, "thinking"),
                             chat_effort(yyjson_obj_get(root, "reasoning_effort"), 75), "<｜deepseek_image｜>", &err,
                             &images, &nimages);
  if (!prompt) die("%s: %s", path, err);
  fputs(prompt, stdout);
  for (int i = 0; i < nimages; i++) {
    yyjson_val *u = yyjson_obj_get(images[i], "image_url");
    const char *url = yyjson_is_str(u) ? yyjson_get_str(u) : yyjson_get_str(yyjson_obj_get(u, "url"));
    fprintf(stderr, "image %s\n", url ? url : "(not an image_url)");
  }
}

// ---- the model's output: reasoning until </think>, content, then DSML tool calls ----

void parser_init(Parser *p, int thinking) {
  memset(p, 0, sizeof *p);
  p->state = thinking ? P_REASON : P_TEXT;
}

static const char *find(const char *s, size_t n, const char *t) {
  size_t m = strlen(t);
  for (size_t i = 0; i + m <= n; i++)
    if (s[i] == t[0] && !memcmp(s + i, t, m)) return s + i;
  return 0;
}

// Bytes at the end of s that may begin t.
static size_t partial(const char *s, size_t n, const char *t) {
  size_t m = strlen(t);
  for (size_t k = m - 1 < n ? m - 1 : n; k > 0; k--)
    if (!memcmp(s + n - k, t, k)) return k;
  return 0;
}

// Valid UTF-8, with invalid bytes as U+FFFD.
static void emit(Parser *p, const char *s, size_t n) {
  Str t = {0};
  for (size_t i = 0; i < n;) {
    unsigned char c = s[i];
    size_t k = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    int ok = k && i + k <= n;
    for (size_t j = 1; ok && j < k; j++) ok = ((unsigned char)s[i + j] >> 6) == 2;
    if (ok) str_put(&t, s + i, k), i += k;
    else str_cat(&t, "\xEF\xBF\xBD"), i++;
  }
  if (t.n) p->text(p->ctx, p->state == P_REASON, t.p, t.n);
  free(t.p);
}

static size_t utf8_tail(const char *s, size_t n) {  // bytes of an unfinished character at the end
  for (size_t k = 1; k <= 3 && k <= n; k++) {
    unsigned char c = s[n - k];
    if ((c >> 6) == 2) continue;
    size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    return len > k ? k : 0;
  }
  return 0;
}

// One DSML invoke: its name and parameters, the arguments as encoding.py's
// decode_dsml_to_arguments writes them (a value that is not JSON is a string).
static int invoke(Parser *p, const char *s, const char *end) {
  const char *q = find(s, end - s, "<｜DSML｜ invoke name=\""), *e;
  if (!q || !(e = find(q, end - q, "\">"))) return 0;
  q += strlen("<｜DSML｜ invoke name=\"");
  Str name = {0}, args = {0};
  str_put(&name, q, e - q);
  str_cat(&args, "{");
  int nparam = 0;
  for (q = e; (q = find(q, end - q, "<｜DSML｜ parameter name=\"")); nparam++) {
    q += strlen("<｜DSML｜ parameter name=\"");
    const char *ke = find(q, end - q, "\" string=\""), *ve, *v;
    if (!ke || !(v = find(ke, end - ke, "\">")) || !(ve = find(v, end - v, "</｜DSML｜ parameter"))) break;
    int string = !memcmp(ke + strlen("\" string=\""), "true", 4);
    v += 2;
    if (nparam) str_cat(&args, ", ");
    json_str(&args, q, ke - q);
    str_cat(&args, ": ");
    yyjson_doc *d = string ? 0 : yyjson_read(v, ve - v, 0);
    if (d) str_put(&args, v, ve - v);
    else json_str(&args, v, ve - v);
    yyjson_doc_free(d);
    q = ve;
  }
  str_cat(&args, "}");
  p->call(p->ctx, name.p, args.p);
  p->calls++;
  free(name.p), free(args.p);
  return 1;
}

static void calls(Parser *p) {
  for (;;) {
    const char *s = p->dsml.p + p->done, *end = p->dsml.p + p->dsml.n;
    const char *ie = find(s, end - s, "</｜DSML｜ invoke>"), *ce = find(s, end - s, "</｜DSML｜ calls>");
    if (ce && (!ie || ce < ie)) {
      p->state = P_END;
      return;
    }
    if (!ie) return;
    if (!invoke(p, s, ie)) {  // not DSML after all: the text as content
      int state = p->state;
      p->state = P_TEXT;
      emit(p, s, ie - s + strlen("</｜DSML｜ invoke>"));
      p->state = state;
    }
    p->done = ie + strlen("</｜DSML｜ invoke>") - p->dsml.p;
  }
}

void parser_feed(Parser *p, const char *s, size_t n) {
  str_put(&p->pend, s, n);
  for (;;) {
    const char *t = p->pend.p;
    size_t len = p->pend.n, skip;
    if (p->state == P_END) {
      p->pend.n = 0;
      return;
    }
    if (p->state == P_CALLS) {
      str_put(&p->dsml, t, len);
      p->pend.n = 0;
      calls(p);
      return;
    }
    const char *think = p->state == P_REASON ? find(t, len, "</think>") : 0, *dsml = find(t, len, CALLS),
               *eos = find(t, len, EOS), *at = think;
    if (dsml && (!at || dsml < at)) at = dsml;
    if (eos && (!at || eos < at)) at = eos;
    if (!at) {  // all but what may begin a marker or a character
      size_t keep = partial(t, len, "\n\n" CALLS), k;
      if ((k = partial(t, len, CALLS)) > keep) keep = k;
      if ((k = partial(t, len, EOS)) > keep) keep = k;
      if (p->state == P_REASON && (k = partial(t, len, "</think>")) > keep) keep = k;
      if ((k = utf8_tail(t, len - keep)) > 0) keep += k;
      emit(p, t, len - keep);
      memmove(p->pend.p, t + len - keep, keep);
      p->pend.n = keep;
      return;
    }
    size_t before = at - t;
    if (at == dsml)  // the blank line before a tool block is not content
      while (before && t[before - 1] == '\n') before--;
    emit(p, t, before);
    skip = at - t + (at == think ? strlen("</think>") : at == dsml ? strlen(CALLS) : strlen(EOS));
    p->state = at == think ? P_TEXT : at == dsml ? P_CALLS : P_END;
    memmove(p->pend.p, t + skip, len - skip);
    p->pend.n = len - skip;
  }
}

int parser_end(Parser *p) {  // the text still held; 0 if a tool block was left open
  if (p->state == P_REASON || p->state == P_TEXT) emit(p, p->pend.p, p->pend.n);
  int open = p->state == P_CALLS;
  free(p->pend.p), free(p->dsml.p);
  p->pend = p->dsml = (Str){0};
  return !open;
}
