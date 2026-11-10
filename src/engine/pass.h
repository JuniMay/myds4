// A pass: a piece of the engine's work that runs on its own stack and gives
// the processor back where it would block, so that one thread can advance
// several passes. Inside a pass, the GPU layer's and the expert store's
// blocking waits yield to the code that resumes the pass.
#pragma once
#include <dispatch/dispatch.h>
#include <stddef.h>

typedef struct Pass Pass;

// What a pass waits for. `ready` says, without blocking, whether the wait is
// over; `block` waits until it is. Both run outside the pass.
typedef struct {
  int (*ready)(void *context);
  void (*block)(void *context);
  void *context;
} PassWait;

// Model-wide memory that belongs to the running pass: copied out when the
// pass yields and back when it resumes.
typedef struct {
  void *address;
  size_t bytes;
} PassVariable;

// A pass that will run body(argument), with a stack of at least stack_bytes
// (the address space only; pages are committed as they are used).
Pass *pass_new(void (*body)(void *), void *argument, size_t stack_bytes);
// The variables' current values are the pass's starting values. While the
// pass runs they hold its values; the resumer's are put back when it yields.
void pass_variables(Pass *pass, const PassVariable *variables, int count);
// Called after the pass's variables are back, before it continues.
void pass_on_resume(Pass *pass, void (*resumed)(void));
void pass_free(Pass *pass);

// Runs the pass until it yields or ends; returns whether it has ended.
int pass_resume(Pass *pass);
// Whether the pass can continue, without blocking.
int pass_ready(const Pass *pass);
// Waits until the pass can continue.
void pass_block(const Pass *pass);
// Runs the pass to its end, waiting at each of its yields.
void pass_run(Pass *pass);

// The pass running now, or null.
Pass *pass_current(void);
// Inside a pass: gives the processor back until the wait is over. A wait
// that is over already does not yield.
void pass_wait(const PassWait *wait);
// The GPU layer's variables that belong to a pass: its signal numbering.
const PassVariable *gpu_pass_variables(int *count);

// Waits for a dispatch group: inside a pass, by yielding.
void pass_wait_group(dispatch_group_t group);
// Called before a pass yields (the GPU layer checks that no command buffer is
// being encoded), when set.
extern void (*pass_before_yield)(void);
