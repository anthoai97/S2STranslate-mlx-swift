import Foundation

public struct SentencePieceHibikiTextTokenDecoder: HibikiTextTokenDecoding {
    private let pieces: [String]

    public init(modelURL: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: modelURL)
        } catch {
            throw HibikiInferenceError.invalidArtifacts("tokenizer unreadable: \(modelURL.lastPathComponent)")
        }
        self.pieces = try Self.parsePieces(from: data)
    }

    init(pieces: [String]) {
        self.pieces = pieces
    }

    public func piece(for token: Int) -> String? {
        guard !HibikiTextTokenContract.isBlankOrPadding(token),
              token >= 0,
              token < pieces.count else {
            return nil
        }
        return HibikiTextTokenContract.normalizeSentencePiece(pieces[token])
    }

    private static func parsePieces(from data: Data) throws -> [String] {
        var reader = ProtobufReader(data)
        var pieces: [String] = []
        while let field = try reader.nextField() {
            if field.number == 1, field.wireType == 2 {
                let message = try reader.lengthDelimited()
                if let piece = try parsePiece(from: message) {
                    pieces.append(piece)
                }
            } else {
                try reader.skip(field)
            }
        }
        guard !pieces.isEmpty else {
            throw HibikiInferenceError.invalidArtifacts("tokenizer contained no sentencepiece entries")
        }
        return pieces
    }

    private static func parsePiece(from data: Data) throws -> String? {
        var reader = ProtobufReader(data)
        while let field = try reader.nextField() {
            if field.number == 1, field.wireType == 2 {
                let bytes = try reader.lengthDelimited()
                return String(data: bytes, encoding: .utf8)
            }
            try reader.skip(field)
        }
        return nil
    }
}

private struct ProtobufField {
    var number: Int
    var wireType: Int
}

private struct ProtobufReader {
    private let bytes: [UInt8]
    private var offset = 0

    init(_ data: Data) {
        self.bytes = Array(data)
    }

    mutating func nextField() throws -> ProtobufField? {
        guard offset < bytes.count else { return nil }
        let key = try readVarint()
        return ProtobufField(number: Int(key >> 3), wireType: Int(key & 0x7))
    }

    mutating func lengthDelimited() throws -> Data {
        let length = Int(try readVarint())
        guard length >= 0, offset + length <= bytes.count else {
            throw HibikiInferenceError.invalidArtifacts("tokenizer protobuf length malformed")
        }
        let start = offset
        offset += length
        return Data(bytes[start..<offset])
    }

    mutating func skip(_ field: ProtobufField) throws {
        switch field.wireType {
        case 0:
            _ = try readVarint()
        case 1:
            try skipBytes(8)
        case 2:
            _ = try lengthDelimited()
        case 5:
            try skipBytes(4)
        default:
            throw HibikiInferenceError.invalidArtifacts("tokenizer protobuf wire type unsupported: \(field.wireType)")
        }
    }

    private mutating func skipBytes(_ count: Int) throws {
        guard offset + count <= bytes.count else {
            throw HibikiInferenceError.invalidArtifacts("tokenizer protobuf field truncated")
        }
        offset += count
    }

    private mutating func readVarint() throws -> UInt64 {
        var shift: UInt64 = 0
        var value: UInt64 = 0
        while offset < bytes.count {
            let byte = bytes[offset]
            offset += 1
            value |= UInt64(byte & 0x7f) << shift
            if byte & 0x80 == 0 {
                return value
            }
            shift += 7
            if shift >= 64 {
                break
            }
        }
        throw HibikiInferenceError.invalidArtifacts("tokenizer protobuf varint malformed")
    }
}
