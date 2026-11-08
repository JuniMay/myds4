// Decode supported audio containers in memory and resample to mono 16 kHz.
#include "myds4.h"
#include "memory.h"
#include <AudioToolbox/AudioToolbox.h>
#include <math.h>
#include <string.h>

typedef struct {
  const uint8_t *data;
  size_t size;
} AudioBytes;
static OSStatus read_audio(void *client, SInt64 position, UInt32 count, void *out, UInt32 *read) {
  AudioBytes *bytes = client;
  if (position < 0 || (uint64_t)position > bytes->size)
    return kAudioFilePositionError;
  size_t remaining = bytes->size - (size_t)position;
  *read = remaining < count ? (UInt32)remaining : count;
  memcpy(out, bytes->data + position, *read);
  return noErr;
}
static SInt64 audio_size(void *client) {
  return (SInt64)((AudioBytes *)client)->size;
}

int audio_open(Media *media, const void *data, size_t size, const char **error) {
  AudioBytes input = {data, size};
  AudioFileID file = NULL;
  ExtAudioFileRef decoder = NULL;
  OSStatus status = AudioFileOpenWithCallbacks(&input, read_audio, NULL, audio_size, NULL, 0, &file);
  if (status) {
    *error = "unsupported or invalid audio container";
    return 0;
  }
  status = ExtAudioFileWrapAudioFileID(file, false, &decoder);
  AudioStreamBasicDescription source = {0};
  SInt64 frames = 0;
  UInt32 bytes = sizeof source;
  if (!status)
    status = ExtAudioFileGetProperty(decoder, kExtAudioFileProperty_FileDataFormat, &bytes, &source);
  bytes = sizeof frames;
  if (!status)
    status = ExtAudioFileGetProperty(decoder, kExtAudioFileProperty_FileLengthFrames, &bytes, &frames);
  double seconds = source.mSampleRate > 0 ? frames / source.mSampleRate : 0;
  if (status || seconds < .01 || seconds > 3600) {
    if (decoder)
      ExtAudioFileDispose(decoder);
    AudioFileClose(file);
    *error = "audio duration must be at least 10 ms and at most one hour";
    return 0;
  }
  AudioStreamBasicDescription client = {.mSampleRate = 16000,
                                        .mFormatID = kAudioFormatLinearPCM,
                                        .mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
                                        .mBytesPerPacket = 4,
                                        .mFramesPerPacket = 1,
                                        .mBytesPerFrame = 4,
                                        .mChannelsPerFrame = 1,
                                        .mBitsPerChannel = 32};
  status = ExtAudioFileSetProperty(decoder, kExtAudioFileProperty_ClientDataFormat, sizeof client, &client);
  size_t capacity = (size_t)ceil(seconds * 16000) + 1024;
  float *samples = status ? NULL : memory_host_alloc(capacity * 4, 0);
  size_t count = 0;
  if (samples)
    while (count < capacity) {
      if (shutdown_requested()) {
        status = kAudioFileUnspecifiedError;
        break;
      }
      UInt32 wanted = (UInt32)(capacity - count > 16384 ? 16384 : capacity - count);
      AudioBufferList output = {
          .mNumberBuffers = 1,
          .mBuffers = {{.mNumberChannels = 1, .mDataByteSize = wanted * 4, .mData = samples + count}}};
      status = ExtAudioFileRead(decoder, &wanted, &output);
      if (status || !wanted)
        break;
      count += wanted;
    }
  ExtAudioFileDispose(decoder);
  AudioFileClose(file);
  if (status || !samples || !count) {
    memory_host_free(samples);
    *error = status ? "audio decoding failed" : "model-memory budget cannot fit the audio samples";
    return 0;
  }
  for (size_t i = 0; i < count; i++)
    if (!isfinite(samples[i])) {
      memory_host_free(samples);
      *error = "audio contains nonfinite samples";
      return 0;
    }
  *media = (Media){.kind = MEDIA_AUDIO,
                   .sample_rate = 16000,
                   .sample_count = (int)count,
                   .samples = samples,
                   .ntokens = ((int)(count / 160) + 7) / 8,
                   .hash = 1469598103934665603ull};
  for (size_t i = 0; i < size; i++)
    media->hash = (media->hash ^ ((const uint8_t *)data)[i]) * 1099511628211ull;
  return 1;
}
