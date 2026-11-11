// Passes on their own stacks (pass.h). Everything runs on the engine thread:
// a pass and the code that resumes it take turns, so nothing races.
#include "pass.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "myds4.h"

struct Pass {
  void *stack, *sp;  // the mapping, and the saved stack pointer while not running
  size_t mapped;
  void (*body)(void *);
  void *argument;
  PassWait wait;     // what it waits for; ready is null when it waits for nothing
  int ended;
  PassVariable *variables;  // copied from pass_variables' list
  int nvariables;
  char *saved, *outer;   // the pass's values while it does not run; the resumer's while it does
  void (*resumed)(void);
};

static Pass *running;
static void *resumer_sp;

// Saves the callee-saved registers and the stack pointer in *save and loads
// them from load: the arm64 calling convention keeps x19-x28, the frame
// pointer, the link register and d8-d15 across a call.
__attribute__((naked)) static void pass_switch(void **save, void *load) {
  __asm__(
      "sub sp, sp, #160\n"
      "stp x19, x20, [sp, #0]\n"
      "stp x21, x22, [sp, #16]\n"
      "stp x23, x24, [sp, #32]\n"
      "stp x25, x26, [sp, #48]\n"
      "stp x27, x28, [sp, #64]\n"
      "stp x29, x30, [sp, #80]\n"
      "stp d8, d9, [sp, #96]\n"
      "stp d10, d11, [sp, #112]\n"
      "stp d12, d13, [sp, #128]\n"
      "stp d14, d15, [sp, #144]\n"
      "mov x9, sp\n"
      "str x9, [x0]\n"
      "mov sp, x1\n"
      "ldp x19, x20, [sp, #0]\n"
      "ldp x21, x22, [sp, #16]\n"
      "ldp x23, x24, [sp, #32]\n"
      "ldp x25, x26, [sp, #48]\n"
      "ldp x27, x28, [sp, #64]\n"
      "ldp x29, x30, [sp, #80]\n"
      "ldp d8, d9, [sp, #96]\n"
      "ldp d10, d11, [sp, #112]\n"
      "ldp d12, d13, [sp, #128]\n"
      "ldp d14, d15, [sp, #144]\n"
      "add sp, sp, #160\n"
      "ret\n");
}

// A new pass's first resume returns here, on the pass's own stack.
static void pass_entry(void) {
  Pass *p = running;
  p->body(p->argument);
  p->ended = 1;
  pass_switch(&p->sp, resumer_sp);
  die("a finished pass was resumed");
}

Pass *pass_new(void (*body)(void *), void *argument, size_t stack_bytes) {
  Pass *p = xmalloc(sizeof *p);
  size_t page = (size_t)getpagesize(), bytes = (stack_bytes + page - 1) / page * page + page;
  void *stack = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (stack == MAP_FAILED) die("cannot map a pass's stack");
  if (mprotect(stack, page, PROT_NONE)) die("cannot protect a pass's stack");  // the guard page
  char *top = (char *)stack + bytes;
  void **frame = (void **)(top - 160);
  memset(frame, 0, 160);
  frame[11] = (void *)pass_entry;  // x30: where the first switch returns
  *p = (Pass){.stack = stack, .mapped = bytes, .sp = frame, .body = body, .argument = argument};
  return p;
}

void pass_variables(Pass *p, const PassVariable *variables, int count) {
  size_t bytes = 0;
  for (int i = 0; i < count; i++) bytes += variables[i].bytes;
  free(p->saved), free(p->outer);
  p->saved = xmalloc(bytes ? bytes : 1), p->outer = xmalloc(bytes ? bytes : 1);
  free(p->variables);
  p->variables = xmalloc((size_t)count * sizeof *p->variables + 1);
  memcpy(p->variables, variables, (size_t)count * sizeof *p->variables);
  p->nvariables = count;
  char *at = p->saved;
  for (int i = 0; i < count; i++) memcpy(at, variables[i].address, variables[i].bytes), at += variables[i].bytes;
}

void pass_on_resume(Pass *p, void (*resumed)(void)) { p->resumed = resumed; }

void pass_free(Pass *p) {
  if (!p) return;
  if (running == p) die("a pass cannot free itself");
  munmap(p->stack, p->mapped);
  free(p->saved), free(p->outer), free(p->variables);
  free(p);
}

int pass_resume(Pass *p) {
  if (running) die("a pass cannot resume another");
  if (p->ended) die("a finished pass was resumed");
  size_t at = 0;
  for (int i = 0; i < p->nvariables; i++) {
    memcpy(p->outer + at, p->variables[i].address, p->variables[i].bytes);
    memcpy(p->variables[i].address, p->saved + at, p->variables[i].bytes);
    at += p->variables[i].bytes;
  }
  if (p->resumed) p->resumed();
  p->wait = (PassWait){0};
  running = p;
  pass_switch(&resumer_sp, p->sp);
  running = NULL;
  at = 0;
  for (int i = 0; i < p->nvariables; i++) {
    memcpy(p->saved + at, p->variables[i].address, p->variables[i].bytes);
    memcpy(p->variables[i].address, p->outer + at, p->variables[i].bytes);
    at += p->variables[i].bytes;
  }
  if (p->resumed) p->resumed();  // what depends on the variables now holds the resumer's values
  return p->ended;
}

int pass_ready(const Pass *p) { return !p->wait.ready || p->wait.ready(p->wait.context); }

void pass_block(const Pass *p) {
  if (p->wait.block) p->wait.block(p->wait.context);
}

void pass_run(Pass *p) {
  while (!pass_resume(p)) pass_block(p);
}

Pass *pass_current(void) { return running; }

void (*pass_before_yield)(void);

void pass_wait(const PassWait *wait) {
  Pass *p = running;
  if (!p) die("pass_wait outside a pass");
  if (wait->ready && wait->ready(wait->context)) return;
  if (pass_before_yield) pass_before_yield();
  p->wait = *wait;
  pass_switch(&p->sp, resumer_sp);
}

static int group_ready(void *group) { return dispatch_group_wait(group, DISPATCH_TIME_NOW) == 0; }
static void group_block(void *group) { dispatch_group_wait(group, DISPATCH_TIME_FOREVER); }

void pass_wait_group(dispatch_group_t group) {
  if (running) pass_wait(&(PassWait){group_ready, group_block, group});
  else group_block(group);
}
