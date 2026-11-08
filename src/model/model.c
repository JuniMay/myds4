// Opening a checkpoint: the family whose model types include its
// config.json's model_type loads it (model.h).
#include <stdlib.h>
#include <string.h>

#include "model.h"

static const ModelOps *const families[] = {&deepseek_ops, &qwen_ops, &flash_ops};

const ModelOps *model_find(const char *dir) {
  char path[1024];
  snprintf(path, sizeof path, "%s/config.json", dir);
  char *text = read_file(path, 0);
  Json *root = json_parse(text);
  const Json *type = json_get(root, "model_type");
  if (!type || type->type != J_STR) die("%s: no model_type", path);
  for (size_t f = 0; f < sizeof families / sizeof *families; f++)
    for (const char *const *t = families[f]->model_types; *t; t++)
      if (!strcmp(*t, type->s)) {
        json_free(root);
        free(text);
        return families[f];
      }
  die("%s: model_type %s is not one this engine runs", path, type->s);
}

Model *model_create(const char *dir) {
  Model *m = xmalloc(sizeof *m);
  *m = (Model){.ops = model_find(dir), .dir = dir};
  memory_account_init(&m->memory, dir);
  memory_slot_init(&m->work, &m->memory, NULL, "model workspace", NULL, NULL);
  m->work.automatic = 1;
  return m;
}

int model_load(Model *m, const Options *o) {
  lock_model();
  memory_scope(&m->work);
  Options options = *o;
  options.family = m->ops == &deepseek_ops ? MODEL_DEEPSEEK : MODEL_QWEN;
  return m->ops->load(m->dir, &options);
}

void model_active(Model *m, int active) {
  memory_scope(&m->work);
  if (m->ops->weight_slot && m->ops->weight_slot()) memory_pin(m->ops->weight_slot(), active);
  if (m->ops->active) m->ops->active(active);
  if (!active) memory_minimum(&m->memory, 0);
}

Model *model_open(const char *dir, const Options *o) {
  Model *m = model_create(dir);
  if (!model_load(m, o)) return NULL;
  model_active(m, 1);
  return m;
}
