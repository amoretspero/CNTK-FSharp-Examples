namespace CNTKExamples

open System
open System.Collections
open System.Collections.Generic
open System.Linq
open System.IO
open System.IO.Compression

type Helper =
    static member DecompressGZip (zipped: byte[]) =
        let gzStream = new GZipStream(new MemoryStream(zipped), CompressionMode.Decompress)
        let decompressedStream = new MemoryStream()
        gzStream.CopyTo(decompressedStream)
        decompressedStream.ToArray()
        