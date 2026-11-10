// Passes on their own stacks (src/engine/pass.c), and what a switch costs.
// Needs no model or GPU.
#include "../../src/engine/pass.h"

#include <stdio.h>
#include <string.h>

#include "myds4.h"

static int checks, failures;
#define CHECK(x) do { checks++; if (!(x)) printf("FAIL: %s\n", #x), failures++; } while (0)

static int counter;
static int always_ready(void *c) { return 1; }
static int flag_ready(void *c) { return *(int *)c; }
static void flag_block(void *c) { *(int *)c = 1; }  // the wait ends when the resumer blocks

typedef struct { int id, yields, flag; char *log; } Args;

// Yields `yields` times, appending its id each time; float registers survive.
static void worker(void *argument) {
  Args *a = argument;
  volatile double kept = a->id * 1.5;
  for (int i = 0; i < a->yields; i++) {
    a->flag = 0;
    pass_wait(&(PassWait){flag_ready, flag_block, &a->flag});
    a->log[strlen(a->log)] = (char)('0' + a->id);
    counter += a->id;
  }
  CHECK(kept == a->id * 1.5);
}

static void waits_for_nothing(void *argument) {
  Pass *self = pass_current();
  pass_wait(&(PassWait){always_ready, NULL, NULL});  // over already: no yield
  CHECK(pass_current() == self);
  *(int *)argument = 1;
}

// A deep use of the stack, to see that it is the pass's own.
static int deep(int depth) {
  volatile char block[64 * 1024];
  block[0] = (char)depth;
  return depth ? deep(depth - 1) + block[0] : block[0];
}
static void deep_body(void *argument) { *(int *)argument = deep(100); }

static int shared;
typedef struct { int value; } Per;

static void uses_variable(void *argument) {
  Per *p = argument;
  CHECK(shared == 0);  // every pass starts from the value it was given
  shared = p->value;
  int never = 0;
  pass_wait(&(PassWait){flag_ready, flag_block, &never});
  CHECK(shared == p->value);  // what the other pass wrote meanwhile is not seen
}

enum { SWITCHES = 2000000 };
static void yielding(void *argument) {
  int flag = 0;
  for (int i = 0; i < SWITCHES; i++) flag = 0, pass_wait(&(PassWait){flag_ready, NULL, &flag});
}

int main(void) {
  // Two passes interleave, each continuing where it yielded.
  char log[32] = {0};
  Args a = {1, 3, 0, log}, b = {2, 3, 0, log};
  Pass *p = pass_new(worker, &a, 1 << 20), *q = pass_new(worker, &b, 1 << 20);
  CHECK(!pass_current());
  for (int round = 0; round < 4; round++) {
    int ended = pass_resume(p);
    pass_block(p);
    ended += pass_resume(q);
    pass_block(q);
    CHECK(ended == (round == 3 ? 2 : 0));
  }
  CHECK(!strcmp(log, "121212"));
  CHECK(counter == 9);
  CHECK(!pass_current());
  pass_free(p), pass_free(q);

  // A wait that is over does not yield, and the pass ends in one resume.
  int done = 0;
  p = pass_new(waits_for_nothing, &done, 1 << 20);
  CHECK(pass_resume(p) == 1);
  CHECK(done == 1);
  pass_free(p);

  // pass_run waits at each yield, and the pass has a stack of its own: deep
  // recursion that would not fit a small stack works on a large one.
  int deep_result = -1;
  p = pass_new(deep_body, &deep_result, 16u << 20);
  pass_run(p);
  CHECK(deep_result > 0);
  pass_free(p);

  // Variables are swapped: each pass sees its own value across a yield.
  shared = 0;
  Per one = {11}, two = {22};
  static const PassVariable variables[] = {{&shared, sizeof shared}};
  p = pass_new(uses_variable, &one, 1 << 20), q = pass_new(uses_variable, &two, 1 << 20);
  pass_variables(p, variables, 1), pass_variables(q, variables, 1);
  CHECK(!pass_resume(p));
  CHECK(!pass_resume(q));
  pass_block(p), pass_block(q);
  CHECK(pass_resume(p) == 1);
  CHECK(pass_resume(q) == 1);
  CHECK(shared == 0);  // the resumer's value is back
  pass_free(p), pass_free(q);

  // What a yield and a resume cost together.
  int flag = 0;
  (void)flag;
  p = pass_new(yielding, NULL, 1 << 20);
  double t0 = now();
  while (!pass_resume(p)) {}
  double seconds = now() - t0;
  pass_free(p);
  printf("passes: %d/%d checks passed; a yield and a resume take %.0f ns\n", checks - failures, checks, seconds / SWITCHES * 1e9);
  return failures != 0;
}
