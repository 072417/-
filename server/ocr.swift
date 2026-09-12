import Foundation
import Vision
import AppKit
let path = CommandLine.arguments[1]
guard let image = NSImage(contentsOfFile: path), let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { exit(2) }
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = ["zh-Hans", "en-US"]
request.usesLanguageCorrection = false
let handler = VNImageRequestHandler(cgImage: cg, options: [:])
do {
 try handler.perform([request])
 let rows: [[String:Any]] = (request.results ?? []).compactMap { o in
  guard let t = o.topCandidates(1).first else { return nil }
  let b = o.boundingBox
  return ["text": t.string, "confidence": t.confidence, "box": [b.minX * Double(cg.width), (1-b.maxY) * Double(cg.height), b.width * Double(cg.width), b.height * Double(cg.height)]]
 }
 let data = try JSONSerialization.data(withJSONObject: rows)
 print(String(data:data,encoding:.utf8)!)
} catch { fputs("OCR failed\n", stderr); exit(3) }
