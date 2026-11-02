// The C host supplies prepared patches. This adapter uses only public Core AI APIs.
import CoreAI
import Dispatch
import Foundation

private enum VisionAIError: Error, CustomStringConvertible {
  case invalid(String)

  var description: String {
    switch self {
    case .invalid(let message): return message
    }
  }
}

private enum VisionAIRope: Sendable {
  case fp16(cos: [Float16], sin: [Float16])
  case fp32(cos: [Float], sin: [Float])
}

private func setMessage(_ text: String, _ buffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt) {
  guard let buffer, capacity > 0 else { return }
  let limit = Int(min(capacity - 1, UInt(Int.max)))
  var index = 0
  for scalar in text.unicodeScalars {
    let bytes = String(scalar).utf8
    if bytes.count > limit - index { break }
    for byte in bytes {
      buffer[index] = CChar(bitPattern: byte)
      index += 1
    }
  }
  buffer[index] = 0
}

// The worker publishes its result before signaling. The caller reads it only
// after waiting on the semaphore, so the result does not cross an unsynchronized access.
private final class Completion<Value: Sendable>: @unchecked Sendable {
  let semaphore = DispatchSemaphore(value: 0)
  var result: Result<Value, Error>?
}

private func waitFor<Value: Sendable>(
  _ operation: @escaping @Sendable () async throws -> Value
) throws -> Value {
  let completion = Completion<Value>()
  Task.detached {
    do {
      completion.result = .success(try await operation())
    } catch {
      completion.result = .failure(error)
    }
    completion.semaphore.signal()
  }
  completion.semaphore.wait()
  return try completion.result!.get()
}

@available(macOS 27.0, *)
private final class VisionAIHandle: @unchecked Sendable {
  let model: AIModel
  let function: InferenceFunction
  let patchCount: Int
  let ropeUsesFloat32: Bool
  let lock = NSLock()

  init(model: AIModel, function: InferenceFunction, patchCount: Int, ropeUsesFloat32: Bool) {
    self.model = model
    self.function = function
    self.patchCount = patchCount
    self.ropeUsesFloat32 = ropeUsesFloat32
  }

  static func load(path: String, entrypoint: String, patchCount: Int) async throws -> VisionAIHandle
  {
    #if VISION_AI_VALIDATE_CPU
      let options: SpecializationOptions = .cpuOnly
    #elseif VISION_AI_VALIDATE_GPU
      let options = SpecializationOptions(preferredComputeUnitKind: .gpu)
    #else
      let options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
    #endif
    let model = try await AIModel(contentsOf: URL(fileURLWithPath: path), options: options)
    guard let function = try model.loadFunction(named: entrypoint) else {
      throw VisionAIError.invalid("Core AI asset has no function named \(entrypoint)")
    }
    let descriptor = function.descriptor
    guard Set(descriptor.inputNames) == Set(["patches", "cos", "sin"]),
      descriptor.stateNames.isEmpty,
      descriptor.outputNames == ["features"]
    else {
      throw VisionAIError.invalid(
        "Core AI function must take patches, cos and sin and return features, without state")
    }
    try validate(
      descriptor.inputDescriptor(of: "patches"), name: "patches",
      shape: [1, 640, 1, patchCount])
    guard let cosine = descriptor.inputDescriptor(of: "cos"),
      case .ndArray(let cosineTensor) = cosine,
      cosineTensor.scalarType == .float16 || cosineTensor.scalarType == .float32
    else {
      throw VisionAIError.invalid("Core AI cos and sin must both use FP16 or both use FP32")
    }
    for name in ["cos", "sin"] {
      try validate(
        descriptor.inputDescriptor(of: name), name: name,
        shape: [1, 32, 1, patchCount], scalarType: cosineTensor.scalarType)
    }
    try validate(
      descriptor.outputDescriptor(of: "features"), name: "features",
      shape: [1, 1024, 1, patchCount])
    return VisionAIHandle(
      model: model, function: function, patchCount: patchCount,
      ropeUsesFloat32: cosineTensor.scalarType == .float32)
  }

  private static func validate(
    _ descriptor: InferenceValue.Descriptor?, name: String,
    shape: [Int], scalarType: NDArray.ScalarType = .float16
  ) throws {
    guard let descriptor, case .ndArray(let tensor) = descriptor,
      !tensor.hasDynamicShape, tensor.shape == shape, tensor.scalarType == scalarType
    else {
      let precision = scalarType == .float32 ? "FP32" : "FP16"
      throw VisionAIError.invalid(
        "Core AI \(name) must be a fixed \(precision) tensor of shape \(shape)")
    }
  }

  private func inputArray<Scalar: BitwiseCopyable>(
    _ values: [Scalar], channels: Int,
    scalarType: NDArray.ScalarType
  ) -> NDArray {
    // This initializer guarantees contiguous row-major storage.
    var array = NDArray(shape: [1, channels, 1, patchCount], scalarType: scalarType)
    array.mutableView(as: Scalar.self).withUnsafeMutablePointer { destination, _, _ in
      values.withUnsafeBufferPointer { source in
        destination.update(from: source.baseAddress!, count: source.count)
      }
    }
    return array
  }

  func run(patches: [Float16], rope: VisionAIRope) async throws -> [Float] {
    let cosine: NDArray
    let sine: NDArray
    switch rope {
    case .fp16(let cos, let sin):
      cosine = inputArray(cos, channels: 32, scalarType: .float16)
      sine = inputArray(sin, channels: 32, scalarType: .float16)
    case .fp32(let cos, let sin):
      cosine = inputArray(cos, channels: 32, scalarType: .float32)
      sine = inputArray(sin, channels: 32, scalarType: .float32)
    }
    let inputs = [
      "patches": inputArray(patches, channels: 640, scalarType: .float16),
      "cos": cosine,
      "sin": sine,
    ]
    var outputs = try await function.run(inputs: inputs)
    guard let value = outputs.remove("features"), let features = value.ndArray else {
      throw VisionAIError.invalid("Core AI returned no features tensor")
    }
    guard features.shape == [1, 1024, 1, patchCount], features.scalarType == .float16 else {
      throw VisionAIError.invalid("Core AI returned features with the wrong shape or scalar type")
    }
    let view = features.view(as: Float16.self)
    var result = [Float](repeating: 0, count: patchCount * 1024)
    if view.interleaveLayout == nil {
      // Strides count elements and may include padding in the returned tensor.
      try view.withUnsafePointer { data, _, strides in
        let channelStride = strides[1]
        let rowStride = strides[3]
        for row in 0..<patchCount {
          let sourceRow = row * rowStride
          let destinationRow = row * 1024
          for channel in 0..<1024 {
            let scalar = Float(data[sourceRow + channel * channelStride])
            guard scalar.isFinite else {
              throw VisionAIError.invalid(
                "Core AI returned a non-finite vision feature at row \(row), channel \(channel)")
            }
            result[destinationRow + channel] = scalar
          }
        }
      }
    } else {
      for row in 0..<patchCount {
        for channel in 0..<1024 {
          let scalar = Float(view[scalarAt: [0, channel, 0, row]])
          guard scalar.isFinite else {
            throw VisionAIError.invalid(
              "Core AI returned a non-finite vision feature at row \(row), channel \(channel)")
          }
          result[row * 1024 + channel] = scalar
        }
      }
    }
    return result
  }
}

private func channelMajor(
  _ source: UnsafePointer<Float>, rows: Int, columns: Int,
  rowStride: Int, offset: Int = 0
) throws -> [Float16] {
  var values = [Float16](repeating: 0, count: rows * columns)
  for column in 0..<columns {
    for row in 0..<rows {
      let value = Float16(source[row * rowStride + offset + column])
      guard value.isFinite else {
        throw VisionAIError.invalid(
          "Vision input cannot be represented as finite FP16 at row \(row), column \(offset + column)"
        )
      }
      values[column * rows + row] = value
    }
  }
  return values
}

private func channelMajorFloat32(
  _ source: UnsafePointer<Float>, rows: Int, columns: Int,
  rowStride: Int, offset: Int = 0
) throws -> [Float] {
  var values = [Float](repeating: 0, count: rows * columns)
  for column in 0..<columns {
    for row in 0..<rows {
      let value = source[row * rowStride + offset + column]
      guard value.isFinite else {
        throw VisionAIError.invalid(
          "Vision input must be finite FP32 at row \(row), column \(offset + column)")
      }
      values[column * rows + row] = value
    }
  }
  return values
}

@c(vision_ai_create)
public func visionAICreate(
  _ path: UnsafePointer<CChar>?, _ entrypoint: UnsafePointer<CChar>?,
  _ patchCount: Int32, _ error: UnsafeMutablePointer<CChar>?,
  _ errorCapacity: UInt
) -> UnsafeMutableRawPointer? {
  setMessage("", error, errorCapacity)
  guard #available(macOS 27.0, *) else {
    setMessage("Core AI vision requires macOS 27 or later", error, errorCapacity)
    return nil
  }
  guard let path, patchCount > 0 else {
    setMessage(
      "Core AI vision needs an asset path and a positive patch count", error, errorCapacity)
    return nil
  }
  let assetPath = String(cString: path)
  let functionName = entrypoint.map { String(cString: $0) } ?? "main"
  do {
    let handle = try waitFor {
      try await VisionAIHandle.load(
        path: assetPath, entrypoint: functionName,
        patchCount: Int(patchCount))
    }
    return Unmanaged.passRetained(handle).toOpaque()
  } catch let failure {
    setMessage(String(describing: failure), error, errorCapacity)
    return nil
  }
}

@c(vision_ai_run)
public func visionAIRun(
  _ pointer: UnsafeMutableRawPointer?, _ patches: UnsafePointer<Float>?,
  _ rope: UnsafePointer<Float>?, _ features: UnsafeMutablePointer<Float>?,
  _ error: UnsafeMutablePointer<CChar>?, _ errorCapacity: UInt
) -> Int32 {
  setMessage("", error, errorCapacity)
  guard #available(macOS 27.0, *) else {
    setMessage("Core AI vision requires macOS 27 or later", error, errorCapacity)
    return 0
  }
  guard let pointer, let patches, let rope, let features else {
    setMessage("Core AI vision needs a valid handle and input/output buffers", error, errorCapacity)
    return 0
  }
  let handle = Unmanaged<VisionAIHandle>.fromOpaque(pointer).takeUnretainedValue()
  handle.lock.lock()
  defer { handle.lock.unlock() }
  do {
    let count = handle.patchCount
    let patchValues = try channelMajor(patches, rows: count, columns: 640, rowStride: 640)
    let ropeValues: VisionAIRope
    if handle.ropeUsesFloat32 {
      ropeValues = .fp32(
        cos: try channelMajorFloat32(rope, rows: count, columns: 32, rowStride: 64),
        sin: try channelMajorFloat32(rope, rows: count, columns: 32, rowStride: 64, offset: 32))
    } else {
      ropeValues = .fp16(
        cos: try channelMajor(rope, rows: count, columns: 32, rowStride: 64),
        sin: try channelMajor(rope, rows: count, columns: 32, rowStride: 64, offset: 32))
    }
    let result = try waitFor {
      try await handle.run(patches: patchValues, rope: ropeValues)
    }
    result.withUnsafeBufferPointer { buffer in
      features.update(from: buffer.baseAddress!, count: buffer.count)
    }
    return 1
  } catch let failure {
    setMessage(String(describing: failure), error, errorCapacity)
    return 0
  }
}

@c(vision_ai_destroy)
public func visionAIDestroy(_ pointer: UnsafeMutableRawPointer?) {
  guard #available(macOS 27.0, *), let pointer else { return }
  Unmanaged<VisionAIHandle>.fromOpaque(pointer).release()
}
